"""
Network class: build units and synapses, run simulation, optionally save checkpoint.
Run from command line or import and call Network(params, rng).run().
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
    cs_us_assembly_patterns,
    compute_W_in_W_out_per_assembly,
)

# EE group block keys for stdp_blocks config (pre_post): CS, US, NS
STDP_BLOCK_KEYS = ['CS_CS', 'CS_US', 'CS_NS', 'US_CS', 'US_US', 'US_NS', 'NS_CS', 'NS_US', 'NS_NS']


def _apply_drive_spike_jitter(indices, times_s, jitter_s, rng):
    """
    Add uniform jitter in ±jitter_s, clip to >=0 (first pulses can become negative
    otherwise), then sort by time for SpikeGeneratorGroup. mergesort keeps stable order.
    """
    if jitter_s <= 0 or len(times_s) == 0:
        return indices, times_s
    times_j = np.asarray(times_s, dtype=np.float64) + rng.uniform(-jitter_s, jitter_s, size=len(times_s))
    times_j = np.maximum(times_j, 0.0)
    order = np.argsort(times_j, kind='mergesort')
    return indices[order], times_j[order]


def _ee_stdp_use_flags(pre_inds, post_inds, cs_inds, us_inds, stdp_blocks):
    """
    Return 1d array of 0/1: 1 if STDP is enabled for that EE synapse, else 0.
    stdp_blocks: None = all True; else dict mapping e.g. 'CS_NS' -> True/False.
    """
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

    set_weights(syn_EE, 'jEE', 0, 0)           # E -> E
    set_weights(syn_EI, 'jEI', 0, nExc)        # I -> E (pre is inh)
    set_weights(syn_IE, 'jIE', nExc, 0)        # E -> I (post is inh)
    set_weights(syn_II, 'jII', nExc, nExc)     # I -> I
    return W


class Network:
    """
    Builds and runs the CS-US recurrent network: neurons, CS/US input, recurrent synapses,
    monitors; runs simulation; stores pre/post weight matrices; optionally saves checkpoint.
    """

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
        self._w_stats_t_list = None
        self._w_stats_w_in = None
        self._w_stats_w_out = None
        self._w_stats_patterns = None
        self._w_stats_i_ee = None
        self._w_stats_j_ee = None

    def _build_neurons(self):
        p = self.params
        rng = self.rng
        use_istdp = p.get('use_istdp', False)
        use_adaptation = p.get('use_adaptation', True)

        if use_adaptation:
            unitModelBase = '''
                dv/dt = (gl * (eLeak - v) - iAdapt +
                         sE - sI + sExt) / Cm +
                         noiseSigma * (Cm / gl)**-0.5 * xi: volt (unless refractory)
                diAdapt/dt = -iAdapt / tauAdapt : amp
                dsE/dt = (-sE + uE) / tauFallE : amp
                duE/dt = -uE / tauRiseE : amp
                dsI/dt = (-sI + uI) / tauFallI : amp
                duI/dt = -uI / tauRiseI : amp
                dsExt/dt = (-sExt + uExt) / tauFallE : amp
                duExt/dt = -uExt / tauRiseE : amp
                eLeak : volt
                vReset : volt
                vThresh : volt
                betaAdapt : amp * second
                gl : siemens
                Cm : farad
            '''
            resetBase = 'v = vReset; iAdapt += betaAdapt / tauAdapt'
        else:
            unitModelBase = '''
                dv/dt = (gl * (eLeak - v) + sE - sI + sExt) / Cm +
                         noiseSigma * (Cm / gl)**-0.5 * xi: volt (unless refractory)
                dsE/dt = (-sE + uE) / tauFallE : amp
                duE/dt = -uE / tauRiseE : amp
                dsI/dt = (-sI + uI) / tauFallI : amp
                duI/dt = -uI / tauRiseI : amp
                dsExt/dt = (-sExt + uExt) / tauFallE : amp
                duExt/dt = -uExt / tauRiseE : amp
                eLeak : volt
                vReset : volt
                vThresh : volt
                betaAdapt : amp * second
                gl : siemens
                Cm : farad
            '''
            resetBase = 'v = vReset'

        use_stdp = p.get('use_stdp', False)
        estdp_traces = ''
        if use_stdp:
            estdp_traces = '''
            dr_stdp/dt = -r_stdp / tau_stdp_plus : 1
            ds_stdp/dt = -s_stdp / tau_stdp_minus : 1
            '''
        if use_istdp:
            unitModelExc = unitModelBase.strip() + '''
            dyE/dt = -yE / tau_y : 1
            ''' + estdp_traces
            unitModelInh = unitModelBase.strip() + '''
            dyI/dt = -yI / tau_y : 1
            '''
            resetCodeExc = resetBase + '; yE += 1'
            resetCodeInh = resetBase + '; yI += 1'
            if use_stdp:
                resetCodeExc += '; r_stdp += 1; s_stdp += 1'
        else:
            unitModelExc = unitModelBase.strip() + estdp_traces if use_stdp else unitModelBase
            unitModelInh = unitModelBase
            resetCodeExc = resetBase + ('; r_stdp += 1; s_stdp += 1' if use_stdp else '')
            resetCodeInh = resetBase

        threshCode = 'v >= vThresh'

        p['nInh'] = int(p['propInh'] * p['nUnits'])
        p['nExc'] = int(p['nUnits'] - p['nInh'])

        # Namespace for neuron ODEs (tau*, noiseSigma, tauAdapt) so they resolve when using explicit Network
        neuron_namespace = {
            'tauRiseE': p['tauRiseExc'],
            'tauFallE': p['tauFallExc'],
            'tauRiseI': p['tauRiseInh'],
            'tauFallI': p['tauFallInh'],
            'tauAdapt': p['adaptTau'],
            'noiseSigma': p['noiseSigma'],
        }
        if use_istdp:
            neuron_namespace['tau_y'] = p['istdp_tau_y']
        if use_stdp:
            neuron_namespace['tau_stdp_plus'] = p['e_stdp_tau_plus']
            neuron_namespace['tau_stdp_minus'] = p['e_stdp_tau_minus']

        self._unitsExc = NeuronGroup(
            N=p['nExc'],
            model=unitModelExc,
            method='euler',
            threshold=threshCode,
            reset=resetCodeExc,
            refractory=p['refrExc'],
            clock=defaultclock,
            namespace=neuron_namespace,
        )
        self._unitsInh = NeuronGroup(
            N=p['nInh'],
            model=unitModelInh,
            method='euler',
            threshold=threshCode,
            reset=resetCodeInh,
            refractory=p['refrInh'],
            clock=defaultclock,
            namespace=neuron_namespace,
        )

        mean_beta = p['betaAdaptExc']
        self._unitsExc.v = p['eLeakExc']
        self._unitsExc.vReset = p['vResetExc']
        self._unitsExc.vThresh = p['vThreshExc']
        self._unitsExc.betaAdapt = mean_beta
        self._unitsExc.eLeak = p['eLeakExc']
        self._unitsExc.Cm = p['membraneCapacitanceExc']
        self._unitsExc.gl = p['gLeakExc']

        self._unitsInh.v = p['eLeakInh']
        self._unitsInh.vReset = p['vResetInh']
        self._unitsInh.vThresh = p['vThreshInh']
        self._unitsInh.betaAdapt = p['betaAdaptInh']
        self._unitsInh.eLeak = p['eLeakInh']
        self._unitsInh.Cm = p['membraneCapacitanceInh']
        self._unitsInh.gl = p['gLeakInh']

        if use_stdp:
            self._unitsExc.r_stdp = 0.0
            self._unitsExc.s_stdp = 0.0

    def _build_cs_us_input(self):
        p = self.params
        n_trials_total = self.n_trials_total

        nCS = max(1, int(round(p['propCS'] * p['nExc'])))
        nUS = max(1, int(round(p['propUS'] * p['nExc'])))
        nUS = min(nUS, p['nExc'] - nCS)
        cs_neuron_inds = np.arange(0, nCS)
        us_neuron_inds = np.arange(nCS, nCS + nUS)
        p['cs_neuron_inds'] = cs_neuron_inds
        p['us_neuron_inds'] = us_neuron_inds

        trial_starts_s = np.array([tr * float(p['trialPeriod'] / second) for tr in range(n_trials_total)])

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
        cs_only_set = set(cs_only_trial_inds)
        p['trial_conditions'] = np.array(['CS' if i in cs_only_set else 'US' for i in range(n_trials_total)])

        cs_indices_src = np.repeat(np.arange(nCS), len(cs_times_s))
        cs_times_expanded = np.tile(cs_times_s, nCS)
        us_indices_src = np.repeat(np.arange(nUS), len(us_times_s))
        us_times_expanded = np.tile(us_times_s, nUS)

        jitter_s = float(p.get('drive_spike_jitter', 0 * ms) / second)
        cs_indices_src, cs_times_expanded = _apply_drive_spike_jitter(
            cs_indices_src, cs_times_expanded, jitter_s, self.rng
        )
        us_indices_src, us_times_expanded = _apply_drive_spike_jitter(
            us_indices_src, us_times_expanded, jitter_s, self.rng
        )

        self._CS_group = SpikeGeneratorGroup(nCS, cs_indices_src, cs_times_expanded * second)
        self._US_group = SpikeGeneratorGroup(nUS, us_indices_src, us_times_expanded * second)

        self._syn_CS = Synapses(self._CS_group, self._unitsExc, on_pre='uExt_post += ' + str(p['spikeInputAmplitude']) + ' * nA')
        self._syn_CS.connect(i=np.arange(nCS), j=cs_neuron_inds)
        self._syn_US = Synapses(self._US_group, self._unitsExc, on_pre='uExt_post += ' + str(p['spikeInputAmplitude']) + ' * nA')
        self._syn_US.connect(i=np.arange(nUS), j=us_neuron_inds)

        # Optional: perturb NS neurons at a given trial and time (for testing)
        self._NS_perturb_group = None
        self._syn_NS_perturb = None
        pert_trial = p.get('ns_perturbation_trial')
        pert_t_s = p.get('ns_perturbation_t_s')
        pert_amp = p.get('ns_perturbation_amplitude')
        if pert_trial is not None and pert_t_s is not None and 0 <= pert_trial < n_trials_total:
            ns_inds = np.array([i for i in range(p['nExc']) if i not in cs_neuron_inds and i not in us_neuron_inds])
            if len(ns_inds) > 0:
                t_pert = trial_starts_s[pert_trial] + float(pert_t_s)
                amp = float(pert_amp if pert_amp is not None else p['spikeInputAmplitude'])
                n_ns = len(ns_inds)
                self._NS_perturb_group = SpikeGeneratorGroup(
                    n_ns,
                    np.arange(n_ns),
                    np.full(n_ns, t_pert) * second,
                )
                self._syn_NS_perturb = Synapses(
                    self._NS_perturb_group,
                    self._unitsExc,
                    on_pre='uExt_post += ' + str(amp) + ' * nA',
                )
                self._syn_NS_perturb.connect(i=np.arange(n_ns), j=ns_inds)

        # Optional: transient kick to random small % of excitatory neurons (to probe up-state initiation / structured spontaneity)
        self._upstate_kick_group = None
        self._syn_upstate_kick = None
        kick_t = p.get('upstate_kick_t_s')
        if kick_t is not None and p['nExc'] > 0:
            rng = self.rng
            fraction = float(p.get('upstate_kick_fraction', 0.02))
            n_kick = max(1, min(p['nExc'], int(round(p['nExc'] * fraction))))
            kick_neurons = rng.choice(p['nExc'], size=n_kick, replace=False)
            amp = float(p.get('upstate_kick_amplitude', p['spikeInputAmplitude']))
            n_pulses = int(p.get('upstate_kick_n_pulses', 1))
            kick_Hz = float(p.get('upstate_kick_Hz', 50))
            if n_pulses <= 1:
                indices = np.arange(n_kick)
                times = np.full(n_kick, kick_t)
            else:
                pulse_times = kick_t + np.arange(n_pulses) / kick_Hz
                indices = np.repeat(np.arange(n_kick), n_pulses)
                times = np.tile(pulse_times, n_kick)
            self._upstate_kick_group = SpikeGeneratorGroup(
                n_kick,
                indices,
                times * second,
            )
            self._syn_upstate_kick = Synapses(
                self._upstate_kick_group,
                self._unitsExc,
                on_pre='uExt_post += ' + str(amp) + ' * nA',
            )
            self._syn_upstate_kick.connect(i=np.arange(n_kick), j=kick_neurons)
            p['upstate_kick_neuron_inds'] = kick_neurons
            p['upstate_kick_n'] = n_kick

    def _build_recurrent_synapses(self):
        p = self.params
        rng = self.rng
        unitsExc = self._unitsExc
        unitsInh = self._unitsInh

        tauRiseEOverMS = p['tauRiseExc'] / ms
        tauRiseIOverMS = p['tauRiseInh'] / ms
        syn_namespace = {'tauRiseEOverMS': tauRiseEOverMS, 'tauRiseIOverMS': tauRiseIOverMS}

        use_homeostatic = p.get('use_homeostatic_norm', False) and p.get('homeostatic_norm_period') is not None
        if p.get('use_stdp', False):
            eqs_EE = '''
                jEE : amp
                use_stdp : 1 (constant)
            '''
            if use_homeostatic:
                eqs_EE = eqs_EE.strip() + '\n                jEE_start : amp (constant)'
            on_pre_EE = '''
                uE_post += jEE / tauRiseEOverMS
                jEE = clip(jEE - use_stdp * e_stdp_A_minus * s_stdp_post, w_min_EE, w_max_EE)
            '''
            on_post_EE = '''
                jEE = clip(jEE + use_stdp * e_stdp_A_plus * r_stdp_pre, w_min_EE, w_max_EE)
            '''
            synapsesEE = Synapses(
                source=unitsExc, target=unitsExc,
                model=eqs_EE, on_pre=on_pre_EE, on_post=on_post_EE,
                namespace=dict(syn_namespace),
            )
            preInds, postInds = adjacency_indices_within(p['nExc'], p['propConnect'], rng)
            synapsesEE.connect(i=preInds, j=postInds)
            synapsesEE.jEE = normal_weights(p['jEE'], len(synapsesEE), p['weightCV'], rng)
            if use_homeostatic:
                synapsesEE.jEE_start = synapsesEE.jEE[:]
            synapsesEE.use_stdp = _ee_stdp_use_flags(
                preInds, postInds,
                p['cs_neuron_inds'], p['us_neuron_inds'],
                p.get('stdp_blocks'),
            )
            synapsesEE.namespace.update({
                'e_stdp_A_plus': p['e_stdp_A_plus'],
                'e_stdp_A_minus': p['e_stdp_A_minus'],
                'w_min_EE': p['w_min_EE'],
                'w_max_EE': p['w_max_EE'],
            })
        else:
            eqs_EE_simple = 'jEE : amp'
            if use_homeostatic:
                eqs_EE_simple += '\njEE_start : amp (constant)'
            synapsesEE = Synapses(
                model=eqs_EE_simple,
                source=unitsExc, target=unitsExc,
                on_pre='uE_post += jEE / tauRiseEOverMS',
                namespace=dict(syn_namespace),
            )
            preInds, postInds = adjacency_indices_within(p['nExc'], p['propConnect'], rng)
            synapsesEE.connect(i=preInds, j=postInds)
            synapsesEE.jEE = normal_weights(p['jEE'], len(synapsesEE), p['weightCV'], rng)
            if use_homeostatic:
                synapsesEE.jEE_start = synapsesEE.jEE[:]

        if p.get('use_istdp', False):
            # iSTDP: presynaptic (I) spike -> jEI += Z*(yE_post - 2*r0*tau_y); postsynaptic (E) spike -> jEI += Z*yI_pre
            two_r0_tau_y = 2.0 * float(p['r0'] / Hz) * float(p['istdp_tau_y'] / second)
            istdp_namespace = dict(syn_namespace)
            istdp_namespace.update({
                'Z_istdp': p['Z'],
                'two_r0_tau_y': two_r0_tau_y,
                'J_EI_min': p['J_EI_min'],
                'J_EI_max': p['J_EI_max'],
            })
            on_pre_EI = '''
                uI_post += jEI / tauRiseIOverMS
                jEI = clip(jEI + Z_istdp * (yE_post - two_r0_tau_y), J_EI_min, J_EI_max)
            '''
            on_post_EI = '''
                jEI = clip(jEI + Z_istdp * yI_pre, J_EI_min, J_EI_max)
            '''
            synapsesEI = Synapses(
                source=unitsInh, target=unitsExc,
                model='jEI : amp',
                on_pre=on_pre_EI, on_post=on_post_EI,
                namespace=istdp_namespace,
            )
        else:
            synapsesEI = Synapses(
                model='jEI: amp',
                source=unitsInh, target=unitsExc,
                on_pre='uI_post += jEI / tauRiseIOverMS',
                namespace=dict(syn_namespace),
            )
        preInds, postInds = adjacency_indices_between(p['nInh'], p['nExc'], p['propConnect'], rng)
        synapsesEI.connect(i=preInds, j=postInds)
        synapsesEI.jEI = normal_weights(p['jEI'], len(synapsesEI), p['weightCV'], rng)

        synapsesIE = Synapses(
            model='jIE: amp',
            source=unitsExc, target=unitsInh,
            on_pre='uE_post += jIE / tauRiseEOverMS',
            namespace=dict(syn_namespace),
        )
        preInds, postInds = adjacency_indices_between(p['nExc'], p['nInh'], p['propConnect'], rng)
        synapsesIE.connect(i=preInds, j=postInds)
        synapsesIE.jIE = normal_weights(p['jIE'], len(synapsesIE), p['weightCV'], rng)

        synapsesII = Synapses(
            model='jII: amp',
            source=unitsInh, target=unitsInh,
            on_pre='uI_post += jII / tauRiseIOverMS',
            namespace=dict(syn_namespace),
        )
        preInds, postInds = adjacency_indices_within(p['nInh'], p['propConnect'], rng)
        synapsesII.connect(i=preInds, j=postInds)
        synapsesII.jII = normal_weights(p['jII'], len(synapsesII), p['weightCV'], rng)

        n_ee, n_ei, n_ie, n_ii = len(synapsesEE), len(synapsesEI), len(synapsesIE), len(synapsesII)
        synapsesEE.delay = ((rng.random(n_ee) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rng.random(n_ei) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rng.random(n_ie) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rng.random(n_ii) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self._synapsesEE = synapsesEE
        self._synapsesEI = synapsesEI
        self._synapsesIE = synapsesIE
        self._synapsesII = synapsesII

    def _build_recurrent_synapses_from_checkpoint(self):
        """Build recurrent synapses from params['weight_matrix_post'] (e.g. from a loaded checkpoint)."""
        p = self.params
        rng = self.rng
        unitsExc = self._unitsExc
        unitsInh = self._unitsInh
        nExc, nInh = p['nExc'], p['nInh']
        W = np.asarray(p['weight_matrix_post'], dtype=float)

        tauRiseEOverMS = p['tauRiseExc'] / ms
        tauRiseIOverMS = p['tauRiseInh'] / ms
        syn_namespace = {'tauRiseEOverMS': tauRiseEOverMS, 'tauRiseIOverMS': tauRiseIOverMS}

        def connect_from_block(W_block):
            """(post_inds, pre_inds, weights) from nonzero entries of W_block (post, pre)."""
            post_inds, pre_inds = np.where(W_block != 0)
            weights = W_block[post_inds, pre_inds]
            return pre_inds, post_inds, weights

        # EE: W[0:nExc, 0:nExc]
        pre_ee, post_ee, w_ee = connect_from_block(W[0:nExc, 0:nExc])
        use_homeostatic = p.get('use_homeostatic_norm', False) and p.get('homeostatic_norm_period') is not None
        if p.get('use_stdp', False):
            eqs_EE = '''
                jEE : amp
                use_stdp : 1 (constant)
            '''
            if use_homeostatic:
                eqs_EE = eqs_EE.strip() + '\n                jEE_start : amp (constant)'
            on_pre_EE = '''
                uE_post += jEE / tauRiseEOverMS
                jEE = clip(jEE - use_stdp * e_stdp_A_minus * s_stdp_post, w_min_EE, w_max_EE)
            '''
            on_post_EE = '''
                jEE = clip(jEE + use_stdp * e_stdp_A_plus * r_stdp_pre, w_min_EE, w_max_EE)
            '''
            synapsesEE = Synapses(
                source=unitsExc, target=unitsExc,
                model=eqs_EE, on_pre=on_pre_EE, on_post=on_post_EE,
                namespace=dict(syn_namespace),
            )
            synapsesEE.connect(i=pre_ee, j=post_ee)
            synapsesEE.jEE = w_ee * amp  # checkpoint stores weights in SI (amperes)
            if use_homeostatic:
                synapsesEE.jEE_start = synapsesEE.jEE[:]
            synapsesEE.use_stdp = _ee_stdp_use_flags(
                pre_ee, post_ee,
                p['cs_neuron_inds'], p['us_neuron_inds'],
                p.get('stdp_blocks'),
            )
            synapsesEE.namespace.update({
                'e_stdp_A_plus': p['e_stdp_A_plus'],
                'e_stdp_A_minus': p['e_stdp_A_minus'],
                'w_min_EE': p['w_min_EE'],
                'w_max_EE': p['w_max_EE'],
            })
        else:
            eqs_EE_simple = 'jEE : amp'
            if use_homeostatic:
                eqs_EE_simple += '\njEE_start : amp (constant)'
            synapsesEE = Synapses(
                model=eqs_EE_simple,
                source=unitsExc, target=unitsExc,
                on_pre='uE_post += jEE / tauRiseEOverMS',
                namespace=dict(syn_namespace),
            )
            synapsesEE.connect(i=pre_ee, j=post_ee)
            synapsesEE.jEE = w_ee * amp  # checkpoint stores weights in SI (amperes)
            if use_homeostatic:
                synapsesEE.jEE_start = synapsesEE.jEE[:]

        # EI: W[0:nExc, nExc:nExc+nInh]
        pre_ei, post_ei, w_ei = connect_from_block(W[0:nExc, nExc : nExc + nInh])
        if p.get('use_istdp', False):
            two_r0_tau_y = 2.0 * float(p['r0'] / Hz) * float(p['istdp_tau_y'] / second)
            istdp_namespace = dict(syn_namespace)
            istdp_namespace.update({
                'Z_istdp': p['Z'],
                'two_r0_tau_y': two_r0_tau_y,
                'J_EI_min': p['J_EI_min'],
                'J_EI_max': p['J_EI_max'],
            })
            on_pre_EI_ck = '''
                uI_post += jEI / tauRiseIOverMS
                jEI = clip(jEI + Z_istdp * (yE_post - two_r0_tau_y), J_EI_min, J_EI_max)
            '''
            on_post_EI_ck = '''
                jEI = clip(jEI + Z_istdp * yI_pre, J_EI_min, J_EI_max)
            '''
            synapsesEI = Synapses(
                source=unitsInh, target=unitsExc,
                model='jEI : amp',
                on_pre=on_pre_EI_ck, on_post=on_post_EI_ck,
                namespace=istdp_namespace,
            )
        else:
            synapsesEI = Synapses(
                model='jEI: amp',
                source=unitsInh, target=unitsExc,
                on_pre='uI_post += jEI / tauRiseIOverMS',
                namespace=dict(syn_namespace),
            )
        synapsesEI.connect(i=pre_ei, j=post_ei)
        synapsesEI.jEI = w_ei * amp  # checkpoint stores weights in SI (amperes)

        # IE: W[nExc:nExc+nInh, 0:nExc]
        pre_ie, post_ie, w_ie = connect_from_block(W[nExc : nExc + nInh, 0:nExc])
        synapsesIE = Synapses(
            model='jIE: amp',
            source=unitsExc, target=unitsInh,
            on_pre='uE_post += jIE / tauRiseEOverMS',
            namespace=dict(syn_namespace),
        )
        synapsesIE.connect(i=pre_ie, j=post_ie)
        synapsesIE.jIE = w_ie * amp  # checkpoint stores weights in SI (amperes)

        # II: W[nExc:nExc+nInh, nExc:nExc+nInh]
        pre_ii, post_ii, w_ii = connect_from_block(W[nExc : nExc + nInh, nExc : nExc + nInh])
        synapsesII = Synapses(
            model='jII: amp',
            source=unitsInh, target=unitsInh,
            on_pre='uI_post += jII / tauRiseIOverMS',
            namespace=dict(syn_namespace),
        )
        synapsesII.connect(i=pre_ii, j=post_ii)
        synapsesII.jII = w_ii * amp  # checkpoint stores weights in SI (amperes)

        # Delays not stored in checkpoint; resample
        n_ee, n_ei, n_ie, n_ii = len(synapsesEE), len(synapsesEI), len(synapsesIE), len(synapsesII)
        synapsesEE.delay = ((rng.random(n_ee) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rng.random(n_ei) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rng.random(n_ie) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rng.random(n_ii) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self._synapsesEE = synapsesEE
        self._synapsesEI = synapsesEI
        self._synapsesIE = synapsesIE
        self._synapsesII = synapsesII

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
        """
        Build network (neurons, CS/US input, recurrent synapses, monitors), run simulation,
        store pre/post weight matrices, optionally save checkpoint.
        If params['load_checkpoint_path'] is set, weights are loaded from that file and
        must match current nExc, nInh; all other params come from the current params dict.
        Returns (params, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh).
        """
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

        # Homeostatic normalization: scale EE weights per post so total input strength
        # sum(jEE) = sum(jEE_start), enabling competition. Fully vectorized (no Python loop over neurons).
        homeostatic_op = None
        if (p.get('use_homeostatic_norm', False) and p.get('homeostatic_norm_period') is not None
                and hasattr(self._synapsesEE, 'jEE_start')):
            period = p['homeostatic_norm_period']
            nExc = p['nExc']
            j_unit = self._synapsesEE.jEE.unit
            w_min = float(p.get('w_min_EE', 0 * pA) / j_unit)
            w_max = float(p.get('w_max_EE', 1e9 * pA) / j_unit)

            def _homeostatic_ee_norm():
                syn = self._synapsesEE

                post_inds = np.asarray(syn.j)
                j_cur = np.asarray(syn.jEE)
                j_start = np.asarray(syn.jEE_start)

                # current and initial totals per postsynaptic neuron
                sum_cur = np.bincount(post_inds, weights=j_cur, minlength=nExc)
                sum_start = np.bincount(post_inds, weights=j_start, minlength=nExc)

                # number of synapses per post neuron
                counts = np.bincount(post_inds, minlength=nExc)

                # amount to subtract per synapse in that row
                delta = np.zeros(nExc)
                np.divide(sum_cur - sum_start, counts, where=counts > 0, out=delta)

                # subtract from each synapse
                new_j = j_cur - delta[post_inds]
                new_j = np.clip(new_j, w_min, w_max)

                syn.jEE[:] = new_j * j_unit

            homeostatic_op = network_operation(dt=period, when='start')(_homeostatic_ee_norm)

        w_stats_op = None
        if (
            p.get('record_ee_w_stats', False)
            and 'cs_neuron_inds' in p
            and 'us_neuron_inds' in p
        ):
            rd = p.get('w_stats_record_dt', 1 * second)
            dt_min = float(defaultclock.dt / second)
            if float(rd / second) < dt_min - 1e-15:
                raise ValueError(
                    'w_stats_record_dt (%.4g s) must be >= defaultclock.dt (%.4g s)'
                    % (float(rd / second), dt_min)
                )
            self._w_stats_patterns = cs_us_assembly_patterns(
                p['nExc'], p['cs_neuron_inds'], p['us_neuron_inds']
            )
            self._w_stats_i_ee = np.asarray(self._synapsesEE.i, dtype=np.int64)
            self._w_stats_j_ee = np.asarray(self._synapsesEE.j, dtype=np.int64)
            self._w_stats_t_list = []
            self._w_stats_w_in = []
            self._w_stats_w_out = []

            def _snap_ee_assembly_weights():
                t_now = float(defaultclock.t / second)
                w = np.asarray(self._synapsesEE.jEE / amp, dtype=np.float64)
                wi, wo = compute_W_in_W_out_per_assembly(
                    w, self._w_stats_i_ee, self._w_stats_j_ee, self._w_stats_patterns
                )
                if self._w_stats_t_list and abs(self._w_stats_t_list[-1] - t_now) < 1e-12:
                    self._w_stats_w_in[-1] = wi
                    self._w_stats_w_out[-1] = wo
                else:
                    self._w_stats_t_list.append(t_now)
                    self._w_stats_w_in.append(wi)
                    self._w_stats_w_out.append(wo)

            _snap_ee_assembly_weights()
            w_stats_op = network_operation(dt=rd, when='end')(_snap_ee_assembly_weights)

        # Brian2's magic run() only collects objects in the current namespace; our objects
        # live on self, so they are never included. Use an explicit Network and add all objects.
        b2_objects = [
            self._unitsExc,
            self._unitsInh,
            self._CS_group,
            self._US_group,
            self._syn_CS,
            self._syn_US,
            self._synapsesEE,
            self._synapsesEI,
            self._synapsesIE,
            self._synapsesII,
            self._spikeMonExc,
            self._spikeMonInh,
            self._stateMonExc,
            self._stateMonInh,
        ]
        if homeostatic_op is not None:
            b2_objects.append(homeostatic_op)
        if w_stats_op is not None:
            b2_objects.append(w_stats_op)
        if self._NS_perturb_group is not None and self._syn_NS_perturb is not None:
            b2_objects.extend([self._NS_perturb_group, self._syn_NS_perturb])
        if self._upstate_kick_group is not None and self._syn_upstate_kick is not None:
            b2_objects.extend([self._upstate_kick_group, self._syn_upstate_kick])
        b2_net = B2Network(*b2_objects)
        b2_net.run(p['duration'], report=p['reportType'], report_period=p['reportPeriod'], profile=p['doProfile'])

        if self._w_stats_t_list is not None:
            dur_s = float(p['duration'] / second)
            last_t = self._w_stats_t_list[-1] if self._w_stats_t_list else -1.0
            if dur_s - last_t > 1e-9:
                w = np.asarray(self._synapsesEE.jEE / amp, dtype=np.float64)
                wi, wo = compute_W_in_W_out_per_assembly(
                    w, self._w_stats_i_ee, self._w_stats_j_ee, self._w_stats_patterns
                )
                self._w_stats_t_list.append(dur_s)
                self._w_stats_w_in.append(wi)
                self._w_stats_w_out.append(wo)
            p['W_in_t'] = np.asarray(self._w_stats_t_list, dtype=np.float64)
            p['W_in_per_assembly_vals'] = np.asarray(self._w_stats_w_in, dtype=np.float64)
            p['W_out_per_assembly_vals'] = np.asarray(self._w_stats_w_out, dtype=np.float64)
            p['W_in_assembly_labels'] = np.array(['CS', 'US'])

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
