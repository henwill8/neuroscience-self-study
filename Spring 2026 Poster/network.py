"""Network build, run, checkpoint. Use Network(params, rng).run().

iSTDP on I→E (optional, params use_istdp): low-pass traces y_istdp on E and I; on E spike
Δg_EI += η·y_I_pre; on I spike Δg_EI += η·(y_E_post − 2·r0·τ_y), clipped to [g_min_EI, g_max_EI]
(Litwin–Kumar & Doiron sim.jl).
"""
from brian2 import *
from brian2.core.network import Network as B2Network
import numpy as np

from config import get_default_params
from plotting import SimpleResults, plot_all_figures
from utils import (
    pulse_times_train,
    adjacency_indices_within,
    adjacency_indices_between,
    normal_weights,
    save_network_checkpoint,
    load_weights_checkpoint,
    cs_us_ns_pattern_matrix,
    compute_W_in_W_out_per_assembly,
)

# EE group block keys for stdp_blocks config (pre_post): CS, US, NS
STDP_BLOCK_KEYS = ['CS_CS', 'CS_US', 'CS_NS', 'US_CS', 'US_US', 'US_NS', 'NS_CS', 'NS_US', 'NS_NS']


# int(...) avoids Relational-in-Mul in Brian's sympy pass for run_regularly.
_EE_VSTDP_LTP_RUN_REG = (
    'gEE = clip(gEE + block_stdp * int(t > stdp_delay) * int(gEE > 0*siemens) * int(v_post > thetaltp_vstdp) * '
    'int(v_lp_vstdp_post > thetaltd_vstdp) * altp_vstdp * vstdp_ltp_dt * x_vstdp_pre * '
    '(v_post - thetaltp_vstdp) * (v_lp_vstdp_post - thetaltd_vstdp), g_min_EE, g_max_EE)'
)


def _configure_ee_voltage_stdp_ltp(synapsesEE, p):
    synapsesEE.namespace['stdp_delay'] = p['stdp_delay']
    synapsesEE.namespace['thetaltp_vstdp'] = p['thetaltp_vstdp']
    synapsesEE.namespace['thetaltd_vstdp'] = p['thetaltd_vstdp']
    synapsesEE.namespace['altp_vstdp'] = p['altp_vstdp']
    synapsesEE.namespace['vstdp_ltp_dt'] = p['dt']
    synapsesEE.namespace['g_min_EE'] = p['g_min_EE']
    synapsesEE.namespace['g_max_EE'] = p['g_max_EE']
    synapsesEE.run_regularly(_EE_VSTDP_LTP_RUN_REG, dt=p['dt'])


def _ee_positive_indegree_per_post(synapsesEE, n_exc):
    """Per postsynaptic E: count of incoming EE synapses with gEE_start > 0."""
    post = np.asarray(synapsesEE.j, dtype=np.int64)
    j0 = np.asarray(synapsesEE.gEE_start / siemens, dtype=np.float64)
    positive = (j0 > 0).astype(np.float64)
    return np.bincount(post, weights=positive, minlength=n_exc).astype(np.int64)


def _use_homeostatic_run_regularly(p):
    return bool(p.get('use_homeostatic_norm', False) and p.get('homeostatic_norm_period') is not None)


def _use_istdp(p):
    """Inhibitory STDP on I→E (sim.jl: trace_istdp, eta, r0, tauy)."""
    return bool(p.get('use_istdp', True))


_HOMEOSTATIC_EE_EXTRA_RUN_REG = '''
                S_ee_in_post = gEE : siemens (summed)
                nee_pos : 1 (constant)
                homeo_apply : 1 (constant)
'''

_HOMEOSTATIC_RUN_REGULARLY_CODE = (
    'gEE = clip(gEE - homeostatic_norm_beta * (S_ee_in_post - S_ee_target_post) / nee_pos * homeo_apply, '
    'g_min_EE, g_max_EE)'
)


def _configure_homeostatic_run_regularly(synapsesEE, units_exc, p):
    n_exc = int(p['nExc'])
    j0 = np.asarray(synapsesEE.gEE_start / siemens, dtype=np.float64)
    posts = np.asarray(synapsesEE.j, dtype=np.int64)
    sum_start = np.bincount(posts, weights=j0, minlength=n_exc)
    units_exc.S_ee_target = sum_start * siemens
    pos = j0 > 0.0
    cpos = np.bincount(posts, weights=pos.astype(np.float64), minlength=n_exc)
    cpos = np.maximum(cpos, 1.0)
    synapsesEE.nee_pos = cpos[posts]
    synapsesEE.homeo_apply = pos.astype(np.float64)
    synapsesEE.namespace['homeostatic_norm_beta'] = float(p.get('homeostatic_norm_beta', 1.0))
    synapsesEE.namespace['g_min_EE'] = p['g_min_EE']
    synapsesEE.namespace['g_max_EE'] = p['g_max_EE']
    # Must run after the 'synapses' slot: S_ee_in_post = gEE (summed) is updated there.
    # Default when='start' reads stale/zero summed input → spurious (S_ee_in - S_ee_target) and weight explosion.
    synapsesEE.run_regularly(
        _HOMEOSTATIC_RUN_REGULARLY_CODE,
        dt=p['homeostatic_norm_period'],
        when='end',
    )


def _ee_stdp_use_flags(pre_inds, post_inds, cs_inds, us_inds, stdp_blocks):
    """Per-EE-synapse 0/1 flags from stdp_blocks (None = all on)."""
    cs_set = set(np.atleast_1d(cs_inds))
    us_set = set(np.atleast_1d(us_inds))
    n = len(pre_inds)
    use = np.ones(n, dtype=np.int32)
    if stdp_blocks is None:
        return use

    def group(neuron_idx):
        if neuron_idx in cs_set:
            return 0
        if neuron_idx in us_set:
            return 1
        return 2

    for k in range(n):
        pre_g = group(pre_inds[k])
        post_g = group(post_inds[k])
        key = STDP_BLOCK_KEYS[pre_g * 3 + post_g]
        use[k] = 1 if stdp_blocks.get(key, True) else 0
    return use


def _pulse_envelope_linear_rise(t_k, t0, t1, ramp_s):
    """
    Per-pulse scalar gate in [0, 1] at times t_k (s): linear rise from 0 at t0 over ramp_s,
    then 1 until t1. If pulse width < ramp_s, rises linearly from 0 at t0 to 1 at t1.
    """
    t0, t1 = float(t0), float(t1)
    if t1 < t0:
        t0, t1 = t1, t0
    W = t1 - t0
    r = float(ramp_s)
    env = np.zeros_like(t_k, dtype=np.float64)
    if W <= 0 or r <= 0:
        env[(t_k >= t0) & (t_k <= t1)] = 1.0
        return env
    if W >= r:
        rise_end = t0 + r
        m_rise = (t_k >= t0) & (t_k < rise_end)
        env[m_rise] = (t_k[m_rise] - t0) / r
        env[(t_k >= rise_end) & (t_k <= t1)] = 1.0
        return env
    m = (t_k >= t0) & (t_k <= t1)
    env[m] = (t_k[m] - t0) / W
    return env


def _scalar_stimulus_gates_from_intervals(intervals_s, duration_s, dt_s, ramp_s=0.0):
    """
    Gate sampled each dt_s: in [0, 1], max over intervals. Each row [t0, t1] in seconds.
    ramp_s == 0: rectangular (1 inside interval). ramp_s > 0: linear onset from 0 at t0.
    """
    intervals_s = np.asarray(intervals_s, dtype=np.float64).reshape(-1, 2)
    n_steps = int(np.ceil(duration_s / dt_s)) + 2
    t_k = np.arange(n_steps, dtype=np.float64) * dt_s
    gate = np.zeros(n_steps, dtype=np.float64)
    for row in intervals_s:
        if row.size < 2:
            continue
        t0, t1 = float(row[0]), float(row[1])
        if float(ramp_s) > 0:
            gate = np.maximum(gate, _pulse_envelope_linear_rise(t_k, t0, t1, float(ramp_s)))
        else:
            if t1 < t0:
                t0, t1 = t1, t0
            gate = np.maximum(gate, ((t_k >= t0) & (t_k <= t1)).astype(np.float64))
    return gate


def _pulse_times_to_conductance_windows(times_s, width_s, t_clip_s):
    """
    Each nominal pulse time becomes [t, min(t + width, t_clip)] for a short sustained conductance.
    times_s: 1d (s); width_s, t_clip_s in seconds.
    """
    times_s = np.asarray(times_s, dtype=np.float64).ravel()
    if times_s.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    t_end = np.minimum(times_s + float(width_s), float(t_clip_s))
    return np.column_stack([times_s, t_end]).astype(np.float64)


def _build_weight_matrix(syn_EE, syn_EI, syn_IE, syn_II, nExc, nInh):
    """
    Build full weight matrix W (nUnits x nUnits), W[post, pre] = weight from pre to post.
    Unconnected pairs are 0. Returns matrix in same units as synapse weights (stripped to float).
    """
    nUnits = nExc + nInh
    W = np.zeros((nUnits, nUnits))

    def set_weights(syn, varname, post_offset=0, pre_offset=0):
        pre = np.asarray(syn.i)
        post = np.asarray(syn.j)
        w = np.asarray(getattr(syn, varname)[:])
        if hasattr(w, 'unit'):
            w = np.asarray(w / w.unit)
        for k in range(len(w)):
            W[post_offset + post[k], pre_offset + pre[k]] = float(w[k])

    set_weights(syn_EE, 'gEE', 0, 0)           # E -> E
    set_weights(syn_EI, 'gEI', 0, nExc)        # I -> E (pre is inh)
    set_weights(syn_IE, 'gIE', nExc, 0)        # E -> I (post is inh)
    set_weights(syn_II, 'gII', nExc, nExc)     # I -> I
    return W


class Network:
    def __init__(self, params, rng):
        self.params = params
        self.rng = rng
        defaultclock.dt = params['dt']
        self.n_trials_total = params['nTrials'] + (1 if params.get('include_CS_only_trial', False) else 0)

        self._unitsExc = None
        self._unitsInh = None
        self._synapsesEE = None
        self._synapsesEI = None
        self._synapsesIE = None
        self._synapsesII = None
        self._spikeMonExc = None
        self._spikeMonInh = None
        self._stateMonExc = None
        self._stateMonInh = None
        self._ee_indegree_per_post = None
        self._CS_group = None
        self._US_group = None
        self._syn_CS = None
        self._syn_US = None

    def _derive_population_sizes(self):
        p = self.params
        p['nInh'] = int(p['propInh'] * p['nUnits'])
        p['nExc'] = int(p['nUnits'] - p['nInh'])

    def _prepare_cs_us_schedule(self):
        """Set cs/us indices, trial metadata, plotting intervals, and sustained scalar gates (if used)."""
        p = self.params
        n_trials_total = self.n_trials_total

        nCS = max(1, int(round(p['propCS'] * p['nExc'])))
        nUS = max(1, int(round(p['propUS'] * p['nExc'])))
        nUS = min(nUS, p['nExc'] - nCS)
        cs_neuron_inds = np.arange(0, nCS)
        us_neuron_inds = np.arange(nCS, nCS + nUS)
        p['cs_neuron_inds'] = cs_neuron_inds
        p['us_neuron_inds'] = us_neuron_inds

        t_pre = float(p.get('pre_first_trial_delay', 0 * second) / second)
        trial_starts_s = t_pre + np.array(
            [tr * float(p['trialPeriod'] / second) for tr in range(n_trials_total)]
        )

        cs_times_s = pulse_times_train(
            trial_starts_s,
            float(p['CS_train_duration'] / second),
            float(p['CS_Hz'] / Hz),
        )
        n_trials_paired = p['nTrials']
        every_n = p.get('cs_only_every_n_trials')
        if every_n is not None and every_n >= 1:
            trial_inds_US = [i for i in range(n_trials_paired) if (i + 1) % every_n != 0]
            trial_starts_for_US = trial_starts_s[trial_inds_US]
        else:
            trial_starts_for_US = trial_starts_s[:n_trials_paired]
        us_times_s = pulse_times_train(
            trial_starts_for_US + float(p['ISI'] / second),
            float(p['US_train_duration'] / second),
            float(p['US_Hz'] / Hz),
        )

        ISI_s = float(p['ISI'] / second)
        cs_only_trial_inds = []
        if every_n is not None and every_n >= 1:
            cs_only_trial_inds.extend([i for i in range(n_trials_paired) if (i + 1) % every_n == 0])
        if p.get('include_CS_only_trial', False) and n_trials_total > n_trials_paired:
            cs_only_trial_inds.append(n_trials_paired)
        p['us_omit_times_s'] = np.array([trial_starts_s[i] + ISI_s for i in cs_only_trial_inds])
        p['trial_starts_s'] = trial_starts_s
        p['trial_duration_s'] = float(p['trialDuration'] / second)
        cs_dur_s = float(p['CS_train_duration'] / second)
        if len(trial_starts_s) > 0:
            p['cs_stim_epoch_intervals_s'] = np.column_stack(
                [trial_starts_s, trial_starts_s + cs_dur_s]
            ).astype(np.float64)
        else:
            p['cs_stim_epoch_intervals_s'] = np.zeros((0, 2), dtype=np.float64)
        us_dur_s = float(p['US_train_duration'] / second)
        if len(trial_starts_for_US) > 0:
            us_t0 = trial_starts_for_US + ISI_s
            p['us_stim_epoch_intervals_s'] = np.column_stack(
                [us_t0, us_t0 + us_dur_s]
            ).astype(np.float64)
        else:
            p['us_stim_epoch_intervals_s'] = np.zeros((0, 2), dtype=np.float64)
        cs_only_set = set(cs_only_trial_inds)
        p['trial_conditions'] = np.array(['CS' if i in cs_only_set else 'US' for i in range(n_trials_total)])

        mode = p.get('cs_us_stimulus_mode', 'sustained')
        dur_s = float(p['duration'] / second)
        dt_s = float(p['dt'] / second)
        if mode == 'sustained':
            w_cs = float(p.get('sustained_input_width_CS', 4 * ms) / second)
            w_us = float(p.get('sustained_input_width_US', 4 * ms) / second)
            ramp_cs_s = float(p.get('sustained_input_ramp_CS', 0 * ms) / second)
            ramp_us_s = float(p.get('sustained_input_ramp_US', 0 * ms) / second)
            p['cs_stim_intervals_s'] = _pulse_times_to_conductance_windows(cs_times_s, w_cs, dur_s)
            p['us_stim_intervals_s'] = _pulse_times_to_conductance_windows(us_times_s, w_us, dur_s)
            p['cs_stim_pulse_times_s'] = np.asarray(cs_times_s, dtype=np.float64).ravel()
            p['us_stim_pulse_times_s'] = np.asarray(us_times_s, dtype=np.float64).ravel()
            p['_sustained_cs_gate_seq'] = _scalar_stimulus_gates_from_intervals(
                p['cs_stim_intervals_s'], dur_s, dt_s, ramp_cs_s
            )
            p['_sustained_us_gate_seq'] = _scalar_stimulus_gates_from_intervals(
                p['us_stim_intervals_s'], dur_s, dt_s, ramp_us_s
            )
        else:
            p['cs_stim_intervals_s'] = p['cs_stim_epoch_intervals_s']
            p['us_stim_intervals_s'] = p['us_stim_epoch_intervals_s']
            p['cs_stim_pulse_times_s'] = np.asarray(cs_times_s, dtype=np.float64).ravel()
            p['us_stim_pulse_times_s'] = np.asarray(us_times_s, dtype=np.float64).ravel()
            p['_sustained_cs_gate_seq'] = None
            p['_sustained_us_gate_seq'] = None

        p['_cs_times_s_for_spikes'] = cs_times_s
        p['_us_times_s_for_spikes'] = us_times_s
        p['_nCS_spike_src'] = nCS
        p['_nUS_spike_src'] = nUS

    def _build_neurons(self):
        p = self.params
        rng = self.rng

        # Dimensionless filters (Julia forward-input scale); ge = (xd-xr)/(taud-taur) * g_ref with τ diff in ms.
        tau_e_n = float((p['tauFallExc'] - p['tauRiseExc']) / ms)
        tau_i_n = float((p['tauFallInh'] - p['tauRiseInh']) / ms)
        g_ref = p['conductance_filter_ref']
        g_ge_scale = p['conductance_ge_scale']
        conductance_filters = '''
                dx_e_rise/dt = -x_e_rise / tau_e_rise : 1
                dx_e_decay/dt = -x_e_decay / tau_e_decay : 1
                dx_i_rise/dt = -x_i_rise / tau_i_rise : 1
                dx_i_decay/dt = -x_i_decay / tau_i_decay : 1
                ge_syn = (x_e_decay - x_e_rise) / tau_e_diff_n * g_ge_scale : siemens
                gi_syn = (x_i_decay - x_i_rise) / tau_i_diff_n * g_ge_scale : siemens
            '''

        # Inhibitory: LIF + conductance input (sim.jl I dynamics).
        istdp_trace_inh = (
            '                dy_istdp/dt = -y_istdp / tau_istdp : 1\n'
            if _use_istdp(p)
            else ''
        )
        unitModelInh = (
            conductance_filters
            + '''
                dv/dt = (gl * (eLeak - v) + ge_syn * (e_rev_E - v) + gi_syn * (e_rev_I - v)) / Cm +
                         noiseSigma * (Cm / gl)**-0.5 * xi: volt (unless refractory)
'''
            + istdp_trace_inh
            + '''                eLeak : volt
                vReset : volt
                vThresh : volt
                e_rev_E : volt
                e_rev_I : volt
                gl : siemens
                Cm : farad
            '''
        )
        resetCodeInh = 'v = vReset' + ('; y_istdp += 1' if _use_istdp(p) else '')

        use_sustained_cs_us = p.get('cs_us_stimulus_mode', 'sustained') == 'sustained'
        istdp_trace_exc = (
            '                dy_istdp/dt = -y_istdp / tau_istdp : 1\n'
            if _use_istdp(p)
            else ''
        )
        stim_add_scale = float(p.get('sustained_input_additive_noise_scale', 0.0))
        sustained_stim_add_noise = (
            ' + sustained_input_additive_noise_scale * noiseSigma * (Cm / gl)**-0.5 * (cs_stim_mask * cs_stim_gate(t) + us_stim_mask * us_stim_gate(t)) * xi_2'
            if use_sustained_cs_us and stim_add_scale != 0.0
            else ''
        )
        eif_dv_core = (
            '''
                dv/dt = ((eLeak - v) + eif_dT * exp((v - vDynT) / eif_dT)) / tau_mem_e +
                         (ge_syn * (e_rev_E - v) + gi_syn * (e_rev_I - v)'''
            + (
                ' + (g_cs_sust * cs_stim_mask * cs_stim_gain * cs_stim_gate(t) + g_us_sust * us_stim_mask * us_stim_gain * us_stim_gate(t)) * (e_rev_E - v)'
                if use_sustained_cs_us
                else ''
            )
            + ''' + gAdapt * (eif_gAdapt_reversal - v)) / Cm'''
            + sustained_stim_add_noise
            + '''
                         + noiseSigma * (Cm / gl)**-0.5 * xi : volt (unless refractory)
                v_dep_adapt = 0.5 * (v - eLeak + abs(v - eLeak)) : volt
                dgAdapt/dt = (eif_gAdapt_subthreshold_drive * v_dep_adapt - gAdapt) / eif_tau_w : siemens
                dvDynT/dt = (v_soft_th0 - vDynT) / tau_vth_dyn : volt
'''
            + istdp_trace_exc
            + '''                eLeak : volt
                vReset : volt
                v_soft_th0 : volt
                eif_dT : volt
                v_peak_eif : volt
                vth_bump : volt
                tau_mem_e : second
                tau_vth_dyn : second
                e_rev_E : volt
                e_rev_I : volt
                gl : siemens
                Cm : farad
                '''
        )
        if use_sustained_cs_us:
            eif_dv_core = (
                eif_dv_core.replace(
                    'Cm : farad',
                    'Cm : farad\n                cs_stim_mask : 1\n                us_stim_mask : 1\n                cs_stim_gain : 1\n                us_stim_gain : 1',
                )
            )
        eif_exc_core = conductance_filters + eif_dv_core
        unitModelExc = eif_exc_core.strip()
        resetCodeExc = 'v = vReset; vDynT += vth_bump; gAdapt += eif_gAdapt_spike'
        if _use_istdp(p):
            resetCodeExc += '; y_istdp += 1'
        threshCodeExc = 'v > v_peak_eif'

        if _use_homeostatic_run_regularly(p):
            unitModelExc = unitModelExc.strip() + '\n                S_ee_in : siemens\n                S_ee_target : siemens\n'

        unitModelExc = unitModelExc.strip() + '''
                du_vstdp/dt = (v - u_vstdp) / tauu_vstdp : volt
                dv_lp_vstdp/dt = (v - v_lp_vstdp) / tauv_vstdp : volt
                dx_vstdp/dt = -x_vstdp / taux_vstdp : hertz
            '''
        resetCodeExc = resetCodeExc + '; x_vstdp += 1/taux_vstdp'

        threshCodeInh = 'v >= vThresh'

        neuron_namespace = {
            'tau_e_diff_n': tau_e_n,
            'tau_i_diff_n': tau_i_n,
            'g_filter_ref': g_ref,
            'g_ge_scale': g_ge_scale,
            'tau_e_rise': p['tauRiseExc'],
            'tau_e_decay': p['tauFallExc'],
            'tau_i_rise': p['tauRiseInh'],
            'tau_i_decay': p['tauFallInh'],
            'noiseSigma': p['noiseSigma'],
            'eif_gAdapt_reversal': p['eif_gAdapt_reversal'],
            'eif_gAdapt_spike': p['eif_gAdapt_spike'],
            'eif_gAdapt_subthreshold_drive': p['eif_gAdapt_subthreshold_drive'],
            'eif_tau_w': p['eif_tau_w'],
            'tauu_vstdp': p['tauu_vstdp'],
            'tauv_vstdp': p['tauv_vstdp'],
            'taux_vstdp': p['taux_vstdp'],
        }
        if use_sustained_cs_us:
            g_cs = p.get('sustained_conductance_CS')
            if g_cs is None:
                g_cs = p.get('spikeInputAmplitude_CS', p['spikeInputAmplitude'])
            g_us = p.get('sustained_conductance_US')
            if g_us is None:
                g_us = p.get('spikeInputAmplitude_US', p['spikeInputAmplitude'])
            neuron_namespace['g_cs_sust'] = g_cs
            neuron_namespace['g_us_sust'] = g_us
            neuron_namespace['cs_stim_gate'] = TimedArray(
                p['_sustained_cs_gate_seq'], dt=p['dt']
            )
            neuron_namespace['us_stim_gate'] = TimedArray(
                p['_sustained_us_gate_seq'], dt=p['dt']
            )
            if stim_add_scale != 0.0:
                neuron_namespace['sustained_input_additive_noise_scale'] = stim_add_scale
        if _use_istdp(p):
            neuron_namespace['tau_istdp'] = p['istdp_tau_y']

        self._unitsExc = NeuronGroup(
            N=p['nExc'],
            model=unitModelExc,
            method='euler',
            threshold=threshCodeExc,
            reset=resetCodeExc,
            refractory=p['refrExc'],
            clock=defaultclock,
            namespace=neuron_namespace,
        )
        self._unitsInh = NeuronGroup(
            N=p['nInh'],
            model=unitModelInh,
            method='euler',
            threshold=threshCodeInh,
            reset=resetCodeInh,
            refractory=p['refrInh'],
            clock=defaultclock,
            namespace=neuron_namespace,
        )

        self._unitsExc.vReset = p['vResetExc']
        self._unitsExc.eLeak = p['eLeakExc']
        self._unitsExc.Cm = p['membraneCapacitanceExc']
        self._unitsExc.gl = p['gLeakExc']
        vr = float(p['vResetExc'] / volt)
        vt0 = float(p['vThreshExc'] / volt)
        self._unitsExc.v = (vr + rng.random(p['nExc']) * (vt0 - vr)) * volt
        self._unitsExc.vDynT = p['vThreshExc']
        self._unitsExc.gAdapt = 0 * siemens
        self._unitsExc.v_soft_th0 = p['vThreshExc']
        self._unitsExc.eif_dT = p['eif_delta_T']
        self._unitsExc.v_peak_eif = p['eif_v_peak']
        self._unitsExc.vth_bump = p['eif_v_th_spike_jump']
        self._unitsExc.tau_mem_e = p['membraneCapacitanceExc'] / p['gLeakExc']
        self._unitsExc.tau_vth_dyn = p['eif_tau_v_th']
        self._unitsExc.u_vstdp = p['vResetExc']
        self._unitsExc.v_lp_vstdp = p['vResetExc']
        self._unitsExc.x_vstdp = 0 * Hz
        self._unitsExc.e_rev_E = p['eRevExcSyn']
        self._unitsExc.e_rev_I = p['eRevInhSyn']
        if use_sustained_cs_us:
            cs_m = np.zeros(p['nExc'], dtype=np.float64)
            us_m = np.zeros(p['nExc'], dtype=np.float64)
            cs_m[np.atleast_1d(p['cs_neuron_inds'])] = 1.0
            us_m[np.atleast_1d(p['us_neuron_inds'])] = 1.0
            self._unitsExc.cs_stim_mask = cs_m
            self._unitsExc.us_stim_mask = us_m
            cs_gain = np.ones(p['nExc'], dtype=np.float64)
            us_gain = np.ones(p['nExc'], dtype=np.float64)
            cv_cs = float(p.get('sustained_input_gain_cv_CS', 0.0))
            cv_us = float(p.get('sustained_input_gain_cv_US', 0.0))
            g_min = float(p.get('sustained_input_gain_min', 0.2))
            cs_inds = np.atleast_1d(p['cs_neuron_inds']).astype(np.int64)
            us_inds = np.atleast_1d(p['us_neuron_inds']).astype(np.int64)
            if cv_cs > 0 and cs_inds.size > 0:
                cs_gain[cs_inds] = np.clip(
                    1.0 + cv_cs * rng.standard_normal(cs_inds.size), g_min, None
                )
            if cv_us > 0 and us_inds.size > 0:
                us_gain[us_inds] = np.clip(
                    1.0 + cv_us * rng.standard_normal(us_inds.size), g_min, None
                )
            self._unitsExc.cs_stim_gain = cs_gain
            self._unitsExc.us_stim_gain = us_gain

        self._unitsInh.v = p['eLeakInh']
        self._unitsInh.vReset = p['vResetInh']
        self._unitsInh.vThresh = p['vThreshInh']
        self._unitsInh.eLeak = p['eLeakInh']
        self._unitsInh.Cm = p['membraneCapacitanceInh']
        self._unitsInh.gl = p['gLeakInh']
        self._unitsInh.e_rev_E = p['eRevExcSyn']
        self._unitsInh.e_rev_I = p['eRevInhSyn']
        if _use_istdp(p):
            self._unitsExc.y_istdp = 0.0
            self._unitsInh.y_istdp = 0.0

    def _build_cs_us_input(self):
        """SpikeGenerator CS/US (pulse_train mode) and optional NS perturb. Sustained mode skips CS/US spikes."""
        p = self.params
        n_trials_total = self.n_trials_total
        trial_starts_s = p['trial_starts_s']
        cs_neuron_inds = p['cs_neuron_inds']
        us_neuron_inds = p['us_neuron_inds']
        nCS = int(p['_nCS_spike_src'])
        nUS = int(p['_nUS_spike_src'])
        cs_times_s = p['_cs_times_s_for_spikes']
        us_times_s = p['_us_times_s_for_spikes']

        if p.get('cs_us_stimulus_mode', 'sustained') == 'pulse_train':
            cs_indices_src = np.repeat(np.arange(nCS), len(cs_times_s))
            cs_times_expanded = np.tile(cs_times_s, nCS)
            us_indices_src = np.repeat(np.arange(nUS), len(us_times_s))
            us_times_expanded = np.tile(us_times_s, nUS)

            self._CS_group = SpikeGeneratorGroup(nCS, cs_indices_src, cs_times_expanded * second)
            self._US_group = SpikeGeneratorGroup(nUS, us_indices_src, us_times_expanded * second)

            g_cs = p.get('spikeInputAmplitude_CS', p['spikeInputAmplitude'])
            g_us = p.get('spikeInputAmplitude_US', p['spikeInputAmplitude'])
            g_ref = p['conductance_filter_ref']
            _on_pre_ext = 'x_e_rise_post += g_stim / g_filter_ref; x_e_decay_post += g_stim / g_filter_ref'
            self._syn_CS = Synapses(
                self._CS_group, self._unitsExc,
                on_pre=_on_pre_ext,
                namespace={'g_stim': g_cs, 'g_filter_ref': g_ref},
            )
            self._syn_CS.connect(i=np.arange(nCS), j=cs_neuron_inds)
            self._syn_US = Synapses(
                self._US_group, self._unitsExc,
                on_pre=_on_pre_ext,
                namespace={'g_stim': g_us, 'g_filter_ref': g_ref},
            )
            self._syn_US.connect(i=np.arange(nUS), j=us_neuron_inds)
        else:
            self._CS_group = None
            self._US_group = None
            self._syn_CS = None
            self._syn_US = None

        self._NS_perturb_group = None
        self._syn_NS_perturb = None
        pert_trial = p.get('ns_perturbation_trial')
        pert_t_s = p.get('ns_perturbation_t_s')
        pert_amp = p.get('ns_perturbation_amplitude')
        if pert_trial is not None and pert_t_s is not None and 0 <= pert_trial < n_trials_total:
            ns_inds = np.array([i for i in range(p['nExc']) if i not in cs_neuron_inds and i not in us_neuron_inds])
            if len(ns_inds) > 0:
                t_pert = trial_starts_s[pert_trial] + float(pert_t_s)
                g_pert = pert_amp if pert_amp is not None else p['spikeInputAmplitude']
                n_ns = len(ns_inds)
                self._NS_perturb_group = SpikeGeneratorGroup(
                    n_ns,
                    np.arange(n_ns),
                    np.full(n_ns, t_pert) * second,
                )
                self._syn_NS_perturb = Synapses(
                    self._NS_perturb_group,
                    self._unitsExc,
                    on_pre=_on_pre_ext,
                    namespace={'g_stim': g_pert, 'g_filter_ref': g_ref},
                )
                self._syn_NS_perturb.connect(i=np.arange(n_ns), j=ns_inds)

    def _build_recurrent_synapses(self):
        p = self.params
        rng = self.rng
        unitsExc = self._unitsExc
        unitsInh = self._unitsInh

        use_homeostatic = _use_homeostatic_run_regularly(p)
        g_ref = p['conductance_filter_ref']
        eqs_EE = '''
                gEE : siemens
                block_stdp : 1 (constant)
            '''
        if use_homeostatic:
            eqs_EE = (
                eqs_EE.strip()
                + '\n                gEE_start : siemens (constant)'
                + _HOMEOSTATIC_EE_EXTRA_RUN_REG
            )
        on_pre_EE = '''
                x_e_rise_post += gEE / g_filter_ref
                x_e_decay_post += gEE / g_filter_ref
                gEE = clip(gEE - block_stdp * int(t > stdp_delay) * int(gEE > 0*siemens) * altd_vstdp * clip(u_vstdp_post - thetaltd_vstdp, 0*volt, 1e9*volt), g_min_EE, g_max_EE)
            '''
        synapsesEE = Synapses(
            source=unitsExc, target=unitsExc,
            model=eqs_EE, on_pre=on_pre_EE,
            namespace={'g_filter_ref': g_ref},
        )
        preInds, postInds = adjacency_indices_within(p['nExc'], p['propConnect'], rng)
        synapsesEE.connect(i=preInds, j=postInds)
        synapsesEE.gEE = normal_weights(p['gEE'], len(synapsesEE), p['weightCV'], rng)
        if use_homeostatic:
            synapsesEE.gEE_start = synapsesEE.gEE[:]
        synapsesEE.block_stdp = _ee_stdp_use_flags(
            preInds, postInds,
            p['cs_neuron_inds'], p['us_neuron_inds'],
            p.get('stdp_blocks'),
        )
        synapsesEE.namespace.update({
            'g_filter_ref': g_ref,
            'g_min_EE': p['g_min_EE'],
            'g_max_EE': p['g_max_EE'],
            'stdp_delay': p['stdp_delay'],
            'altd_vstdp': p['altd_vstdp'],
            'thetaltd_vstdp': p['thetaltd_vstdp'],
        })
        _configure_ee_voltage_stdp_ltp(synapsesEE, p)

        use_istdp = _use_istdp(p)
        if use_istdp:
            two_r0_tau_y = 2 * float(p['istdp_r0'] / Hz) * float(p['istdp_tau_y'] / second)
            ei_model = '''
                gEI : siemens
                plasticity_ei : 1 (constant)
            '''
            ei_on_pre = '''
                x_i_rise_post += gEI / g_filter_ref
                x_i_decay_post += gEI / g_filter_ref
                gEI = clip(gEI + plasticity_ei * int(t > stdp_delay) * int(gEI > 0*siemens) * Z_istdp * (y_istdp_post - two_r0_tau_y), g_min_EI, g_max_EI)
                '''
            ei_on_post = '''
                gEI = clip(gEI + plasticity_ei * int(t > stdp_delay) * int(gEI > 0*siemens) * Z_istdp * y_istdp_pre, g_min_EI, g_max_EI)
                '''
            synapsesEI = Synapses(
                source=unitsInh, target=unitsExc,
                model=ei_model,
                on_pre=ei_on_pre,
                on_post=ei_on_post,
                namespace={
                    'g_filter_ref': g_ref,
                    'Z_istdp': p['istdp_eta'],
                    'two_r0_tau_y': two_r0_tau_y,
                    'g_min_EI': p['g_min_EI'],
                    'g_max_EI': p['g_max_EI'],
                    'stdp_delay': p['stdp_delay'],
                },
            )
        else:
            synapsesEI = Synapses(
                model='gEI: siemens',
                source=unitsInh, target=unitsExc,
                on_pre='''
                    x_i_rise_post += gEI / g_filter_ref
                    x_i_decay_post += gEI / g_filter_ref
                    ''',
                namespace={'g_filter_ref': g_ref},
            )
        preInds, postInds = adjacency_indices_between(p['nInh'], p['nExc'], p['propConnect'], rng)
        synapsesEI.connect(i=preInds, j=postInds)
        synapsesEI.gEI = normal_weights(p['gEI'], len(synapsesEI), p['weightCV'], rng)
        if use_istdp:
            synapsesEI.plasticity_ei = 1

        synapsesIE = Synapses(
            model='gIE: siemens',
            source=unitsExc, target=unitsInh,
            on_pre='''
                x_e_rise_post += gIE / g_filter_ref
                x_e_decay_post += gIE / g_filter_ref
                ''',
            namespace={'g_filter_ref': g_ref},
        )
        preInds, postInds = adjacency_indices_between(p['nExc'], p['nInh'], p['propConnect'], rng)
        synapsesIE.connect(i=preInds, j=postInds)
        synapsesIE.gIE = normal_weights(p['gIE'], len(synapsesIE), p['weightCV'], rng)

        synapsesII = Synapses(
            model='gII: siemens',
            source=unitsInh, target=unitsInh,
            on_pre='''
                x_i_rise_post += gII / g_filter_ref
                x_i_decay_post += gII / g_filter_ref
                ''',
            namespace={'g_filter_ref': g_ref},
        )
        preInds, postInds = adjacency_indices_within(p['nInh'], p['propConnect'], rng)
        synapsesII.connect(i=preInds, j=postInds)
        synapsesII.gII = normal_weights(p['gII'], len(synapsesII), p['weightCV'], rng)

        n_ee, n_ei, n_ie, n_ii = len(synapsesEE), len(synapsesEI), len(synapsesIE), len(synapsesII)
        synapsesEE.delay = ((rng.random(n_ee) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rng.random(n_ei) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rng.random(n_ie) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rng.random(n_ii) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt

        if use_homeostatic:
            _configure_homeostatic_run_regularly(synapsesEE, unitsExc, p)

        self._synapsesEE = synapsesEE
        self._synapsesEI = synapsesEI
        self._synapsesIE = synapsesIE
        self._synapsesII = synapsesII

        if use_homeostatic:
            self._ee_indegree_per_post = _ee_positive_indegree_per_post(synapsesEE, p['nExc'])

    def _build_recurrent_synapses_from_checkpoint(self):
        """Build recurrent synapses from params['weight_matrix_post'] (e.g. from a loaded checkpoint)."""
        p = self.params
        rng = self.rng
        unitsExc = self._unitsExc
        unitsInh = self._unitsInh
        nExc, nInh = p['nExc'], p['nInh']
        W = np.asarray(p['weight_matrix_post'], dtype=float)

        def connect_from_block(W_block):
            """(post_inds, pre_inds, weights) from nonzero entries of W_block (post, pre)."""
            post_inds, pre_inds = np.where(W_block != 0)
            weights = W_block[post_inds, pre_inds]
            return pre_inds, post_inds, weights

        # EE: W[0:nExc, 0:nExc]
        pre_ee, post_ee, w_ee = connect_from_block(W[0:nExc, 0:nExc])
        use_homeostatic = _use_homeostatic_run_regularly(p)
        g_ref = p['conductance_filter_ref']
        eqs_EE = '''
                gEE : siemens
                block_stdp : 1 (constant)
            '''
        if use_homeostatic:
            eqs_EE = (
                eqs_EE.strip()
                + '\n                gEE_start : siemens (constant)'
                + _HOMEOSTATIC_EE_EXTRA_RUN_REG
            )
        on_pre_EE = '''
                x_e_rise_post += gEE / g_filter_ref
                x_e_decay_post += gEE / g_filter_ref
                gEE = clip(gEE - block_stdp * int(t > stdp_delay) * int(gEE > 0*siemens) * altd_vstdp * clip(u_vstdp_post - thetaltd_vstdp, 0*volt, 1e9*volt), g_min_EE, g_max_EE)
            '''
        synapsesEE = Synapses(
            source=unitsExc, target=unitsExc,
            model=eqs_EE, on_pre=on_pre_EE,
            namespace={'g_filter_ref': g_ref},
        )
        synapsesEE.connect(i=pre_ee, j=post_ee)
        synapsesEE.gEE = w_ee * siemens
        if use_homeostatic:
            synapsesEE.gEE_start = synapsesEE.gEE[:]
        synapsesEE.block_stdp = _ee_stdp_use_flags(
            pre_ee, post_ee,
            p['cs_neuron_inds'], p['us_neuron_inds'],
            p.get('stdp_blocks'),
        )
        synapsesEE.namespace.update({
            'g_filter_ref': g_ref,
            'g_min_EE': p['g_min_EE'],
            'g_max_EE': p['g_max_EE'],
            'stdp_delay': p['stdp_delay'],
            'altd_vstdp': p['altd_vstdp'],
            'thetaltd_vstdp': p['thetaltd_vstdp'],
        })
        _configure_ee_voltage_stdp_ltp(synapsesEE, p)

        # EI: W[0:nExc, nExc:nExc+nInh]
        pre_ei, post_ei, w_ei = connect_from_block(W[0:nExc, nExc : nExc + nInh])
        use_istdp = _use_istdp(p)
        if use_istdp:
            two_r0_tau_y = 2 * float(p['istdp_r0'] / Hz) * float(p['istdp_tau_y'] / second)
            ei_model = '''
                gEI : siemens
                plasticity_ei : 1 (constant)
            '''
            ei_on_pre = '''
                x_i_rise_post += gEI / g_filter_ref
                x_i_decay_post += gEI / g_filter_ref
                gEI = clip(gEI + plasticity_ei * int(t > stdp_delay) * int(gEI > 0*siemens) * Z_istdp * (y_istdp_post - two_r0_tau_y), g_min_EI, g_max_EI)
                '''
            ei_on_post = '''
                gEI = clip(gEI + plasticity_ei * int(t > stdp_delay) * int(gEI > 0*siemens) * Z_istdp * y_istdp_pre, g_min_EI, g_max_EI)
                '''
            synapsesEI = Synapses(
                source=unitsInh, target=unitsExc,
                model=ei_model,
                on_pre=ei_on_pre,
                on_post=ei_on_post,
                namespace={
                    'g_filter_ref': g_ref,
                    'Z_istdp': p['istdp_eta'],
                    'two_r0_tau_y': two_r0_tau_y,
                    'g_min_EI': p['g_min_EI'],
                    'g_max_EI': p['g_max_EI'],
                    'stdp_delay': p['stdp_delay'],
                },
            )
        else:
            synapsesEI = Synapses(
                model='gEI: siemens',
                source=unitsInh, target=unitsExc,
                on_pre='''
                    x_i_rise_post += gEI / g_filter_ref
                    x_i_decay_post += gEI / g_filter_ref
                    ''',
                namespace={'g_filter_ref': g_ref},
            )
        synapsesEI.connect(i=pre_ei, j=post_ei)
        synapsesEI.gEI = w_ei * siemens
        if use_istdp:
            synapsesEI.plasticity_ei = 1

        # IE: W[nExc:nExc+nInh, 0:nExc]
        pre_ie, post_ie, w_ie = connect_from_block(W[nExc : nExc + nInh, 0:nExc])
        synapsesIE = Synapses(
            model='gIE: siemens',
            source=unitsExc, target=unitsInh,
            on_pre='''
                x_e_rise_post += gIE / g_filter_ref
                x_e_decay_post += gIE / g_filter_ref
                ''',
            namespace={'g_filter_ref': g_ref},
        )
        synapsesIE.connect(i=pre_ie, j=post_ie)
        synapsesIE.gIE = w_ie * siemens

        # II: W[nExc:nExc+nInh, nExc:nExc+nInh]
        pre_ii, post_ii, w_ii = connect_from_block(W[nExc : nExc + nInh, nExc : nExc + nInh])
        synapsesII = Synapses(
            model='gII: siemens',
            source=unitsInh, target=unitsInh,
            on_pre='''
                x_i_rise_post += gII / g_filter_ref
                x_i_decay_post += gII / g_filter_ref
                ''',
            namespace={'g_filter_ref': g_ref},
        )
        synapsesII.connect(i=pre_ii, j=post_ii)
        synapsesII.gII = w_ii * siemens

        # Delays not stored in checkpoint; resample
        n_ee, n_ei, n_ie, n_ii = len(synapsesEE), len(synapsesEI), len(synapsesIE), len(synapsesII)
        synapsesEE.delay = ((rng.random(n_ee) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rng.random(n_ei) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rng.random(n_ie) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rng.random(n_ii) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt

        if use_homeostatic:
            _configure_homeostatic_run_regularly(synapsesEE, unitsExc, p)

        self._synapsesEE = synapsesEE
        self._synapsesEI = synapsesEI
        self._synapsesIE = synapsesIE
        self._synapsesII = synapsesII

        if use_homeostatic:
            self._ee_indegree_per_post = _ee_positive_indegree_per_post(synapsesEE, nExc)

    def _build_monitors(self):
        p = self.params
        self._spikeMonExc = SpikeMonitor(self._unitsExc)
        self._spikeMonInh = SpikeMonitor(self._unitsInh)

        n_rec = p.get('n_record_voltage')
        if n_rec is None:
            record_exc = True
            record_inh = True
        else:
            n_re = min(int(n_rec), p['nExc'])
            n_ri = min(int(n_rec), p['nInh'])
            record_exc = np.linspace(0, p['nExc'] - 1, n_re, dtype=int)
            record_inh = np.linspace(0, p['nInh'] - 1, n_ri, dtype=int)
        p['record_voltage_exc_inds'] = np.arange(p['nExc']) if record_exc is True else np.asarray(record_exc)
        p['record_voltage_inh_inds'] = np.arange(p['nInh']) if record_inh is True else np.asarray(record_inh)
        self._stateMonExc = StateMonitor(self._unitsExc, 'v', record=record_exc)
        self._stateMonInh = StateMonitor(self._unitsInh, 'v', record=record_inh)

    def run(self):
        """Build, run, fill weight_matrix_post and optional checkpoint. Returns monitors tuple."""
        self._derive_population_sizes()
        self._prepare_cs_us_schedule()
        self._build_neurons()
        self._build_cs_us_input()

        p = self.params
        load_path = p.get('load_checkpoint_path')
        if load_path:
            w_post, ckpt_nExc, ckpt_nInh = load_weights_checkpoint(load_path)
            if ckpt_nExc != p['nExc'] or ckpt_nInh != p['nInh']:
                raise ValueError(
                    "Checkpoint network size (nExc=%d, nInh=%d) does not match current params (nExc=%d, nInh=%d)."
                    % (ckpt_nExc, ckpt_nInh, p['nExc'], p['nInh'])
                )
            p['weight_matrix_post'] = w_post
            p['weight_matrix_pre'] = np.asarray(w_post, dtype=float).copy()
            p['load_checkpoint_path'] = None  # so saved checkpoint doesn't re-load this path
            self._build_recurrent_synapses_from_checkpoint()
        else:
            self._build_recurrent_synapses()
            p['weight_matrix_pre'] = _build_weight_matrix(
                self._synapsesEE, self._synapsesEI, self._synapsesIE, self._synapsesII,
                p['nExc'], p['nInh'],
            )

        self._build_monitors()

        # W_in / W_out: chunked runs + host snapshots between chunks.
        self._w_stats_t_list = None
        self._w_stats_chunked_run = False
        record_w_dt = p.get('w_stats_record_dt', 1 * second)
        if (
            p.get('record_ee_w_stats', False)
            and 'cs_neuron_inds' in p
            and 'us_neuron_inds' in p
        ):
            self._w_stats_patterns = cs_us_ns_pattern_matrix(
                p['nExc'], p['cs_neuron_inds'], p['us_neuron_inds']
            )
            self._w_stats_i_ee = np.asarray(self._synapsesEE.i, dtype=np.int64)
            self._w_stats_j_ee = np.asarray(self._synapsesEE.j, dtype=np.int64)
            self._w_stats_t_list = []
            self._w_stats_w_in = []
            self._w_stats_w_out = []
            self._w_stats_chunked_run = True
            if float(record_w_dt / second) < float(defaultclock.dt / second):
                raise ValueError(
                    'w_stats_record_dt (%.4g s) must be >= defaultclock.dt (%.4g s)'
                    % (float(record_w_dt / second), float(defaultclock.dt / second))
                )

        b2_objects = [
            self._unitsExc,
            self._unitsInh,
            self._synapsesEE,
            self._synapsesEI,
            self._synapsesIE,
            self._synapsesII,
            self._spikeMonExc,
            self._spikeMonInh,
            self._stateMonExc,
            self._stateMonInh,
        ]
        if self._CS_group is not None and self._syn_CS is not None:
            b2_objects.extend([self._CS_group, self._US_group, self._syn_CS, self._syn_US])
        if self._NS_perturb_group is not None and self._syn_NS_perturb is not None:
            b2_objects.extend([self._NS_perturb_group, self._syn_NS_perturb])
        b2_net = B2Network(*b2_objects)

        def _snap_ee_w_stats():
            g = np.asarray(self._synapsesEE.gEE / siemens, dtype=np.float64)
            wi, wo = compute_W_in_W_out_per_assembly(
                g, self._w_stats_i_ee, self._w_stats_j_ee, self._w_stats_patterns
            )
            self._w_stats_t_list.append(float(defaultclock.t / second))
            self._w_stats_w_in.append(wi)
            self._w_stats_w_out.append(wo)

        if self._w_stats_chunked_run:
            total_s = float(p['duration'] / second)
            rd_s = float(record_w_dt / second)
            dt_s = float(defaultclock.dt / second)
            _snap_ee_w_stats()
            rem_s = total_s
            while rem_s > 1e-15:
                step_s = min(rd_s, rem_s)
                step = step_s * second
                rep_type = p['reportType']
                rep_per = p['reportPeriod']
                if rep_type == 'text':
                    rep_per = min(rep_per, step)
                    if float(rep_per / second) < dt_s:
                        rep_per = defaultclock.dt
                is_final_segment = abs(step_s - rem_s) <= 1e-12
                b2_net.run(
                    step,
                    report=rep_type,
                    report_period=rep_per,
                    profile=p['doProfile'] and is_final_segment,
                )
                rem_s -= step_s
                _snap_ee_w_stats()
        else:
            b2_net.run(p['duration'], report=p['reportType'], report_period=p['reportPeriod'], profile=p['doProfile'])

        if getattr(self, '_w_stats_t_list', None) is not None and len(self._w_stats_t_list) > 0:
            p['w_stats_t'] = np.asarray(self._w_stats_t_list, dtype=np.float64)
            p['w_in_CS_US_NS'] = np.asarray(self._w_stats_w_in, dtype=np.float64)
            p['w_out_CS_US_NS'] = np.asarray(self._w_stats_w_out, dtype=np.float64)
        else:
            p['w_stats_t'] = np.array([], dtype=np.float64)
            p['w_in_CS_US_NS'] = np.zeros((0, 3), dtype=np.float64)
            p['w_out_CS_US_NS'] = np.zeros((0, 3), dtype=np.float64)

        p['weight_matrix_post'] = _build_weight_matrix(
            self._synapsesEE, self._synapsesEI, self._synapsesIE, self._synapsesII,
            p['nExc'], p['nInh'],
        )

        if p.get('save_checkpoint', False):
            save_network_checkpoint(p['checkpoint_path'], p)

        return (
            p,
            self._spikeMonExc,
            self._spikeMonInh,
            self._stateMonExc,
            self._stateMonInh,
        )


def main():
    """Default entry point: run network with default params and show all figures.
    To load weights from a checkpoint, set params['load_checkpoint_path'] or pass the path as first sys.argv."""
    import sys
    rng_seed = 42
    seed(rng_seed)
    np.random.seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    params = get_default_params()
    if len(sys.argv) > 1:
        params['load_checkpoint_path'] = sys.argv[1]
    net = Network(params, rng)
    params, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh = net.run()

    results = SimpleResults(
        spikeMonExc,
        spikeMonInh,
        stateMonExc,
        stateMonInh,
        params,
    )
    plot_all_figures(results, show=True)


if __name__ == '__main__':
    main()
