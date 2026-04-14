"""
Default network parameters. Call get_default_params() for a copy with derived fields set.

Synaptic weights are conductances (siemens); recurrent drive matches Litwin–Kumar & Doiron sim.jl
(difference-of-exponentials E/I filters, ge*(E_e-v)/C, gi*(E_i-v)/C). E→E vSTDP uses the Methods
values A_LTD = 0.0008 pA·mV⁻¹ and A_LTP = 0.0014 pA·mV⁻² (mapped to siemens updates as in sim.jl).
I→E weights support iSTDP as in sim.jl (use_istdp, istdp_tau_y, istdp_eta, istdp_r0, g_min_EI, g_max_EI).
"""
from brian2 import *
from utils import trial_duration


def derive_trial_params(params):
    params['trialDuration'] = trial_duration(params) * second
    params['trialPeriod'] = params['trialDuration'] + params['interTrialInterval']
    n_trials_total = params['nTrials'] + (1 if params.get('include_CS_only_trial', False) else 0)
    pre_first = float(params.get('pre_first_trial_delay', 0 * second) / second)
    params['duration'] = (
        pre_first
        + n_trials_total * float(params['trialPeriod'] / second)
        - float(params['interTrialInterval'] / second) + 0.5
    ) * second


def get_default_params():
    params = {
        'dt': 0.1 * ms,
        'reportType': 'stdout',
        'reportPeriod': 10 * second,
        'doProfile': True,

        'nTrials': 1,
        'ISI': 360 * ms,
        'propCS': 0.05,
        'propUS': 0.05,
        'interTrialInterval': 1 * second,
        # Baseline / setup time before trial 0 CS onset (added to total simulation duration).
        'pre_first_trial_delay': 10 * ms,
        'include_CS_only_trial': True,
        'cs_only_every_n_trials': None,
        'CS_train_duration': 440 * ms,
        'CS_Hz': 25 * Hz,
        'US_train_duration': 80 * ms,
        'US_Hz': 50 * Hz,
        # CS/US: conductance step per input spike (Julia jex=1.78 in weight units; our pulse train needs
        # larger nS/step when conductance_ge_scale is subcritical — tune with conductance_ge_scale).
        'spikeInputAmplitude': 8 * nS,
        # Feedforward CS/US: SpikeGenerator → E synapses at CS_Hz / US_Hz (see network._build_cs_us_input).
        'spikeInputAmplitude_CS': 200 * nS,
        'spikeInputAmplitude_US': 200 * nS,
        # Per-neuron Gaussian jitter (s) on each input spike time. Upper-clipped to train end only
        # (no floor at train onset) so the first pulse can fire earlier and stay desynchronized.
        # 0 = synchronized pulses across the group. Use enough pre_first_trial_delay if times must stay >= 0.
        'CS_input_jitter_std': 5 * ms,
        'US_input_jitter_std': 5 * ms,

        'nUnits': 2e3,
        'propInh': 0.20,
        'propConnect': 0.25,

        # E: EIF; C and gL chosen so C/gL = 20 ms like sim.jl (taue=20, C=300, g=15 in paper units).
        'eLeakExc': -70 * mV,
        'vResetExc': -60 * mV,
        'vThreshExc': -52 * mV,
        'refrExc': 1 * ms,
        'membraneCapacitanceExc': 300 * pF,
        'gLeakExc': 15 * nS,
        'eif_delta_T': 2 * mV,
        'eif_v_peak': 20 * mV,
        'eif_v_th_spike_jump': 3 * mV,
        'eif_tau_v_th': 30 * ms,
        # Slow adaptation conductance gAdapt: I_adapt = gAdapt * (E_adapt - v) / Cm (hyperpolarizing if E_adapt < v).
        # dgAdapt/dt = (subthreshold_drive * clip(v - E_leak, 0, dep_clip_hi) - gAdapt) / tau_adapt; reset adds eif_gAdapt_spike.
        #   subthreshold_drive scales (v - E_leak)+, not gAdapt. If 0, dg/dt = -gAdapt/tau — spike jumps still adapt via eif_gAdapt_spike.
        #   dep_clip_hi: upper cap on (v - E_leak) for the drive (mV above rest); ~peak - leak is enough headroom.
        'eif_gAdapt_reversal': -85 * mV,
        'eif_gAdapt_spike': 30 * nS,
        'eif_gAdapt_subthreshold_drive': 0.08 * nS / mV,
        'eif_tau_w': 150 * ms,

        # I: LIF; same C, gL for taui=20 ms in sim.jl; vleaki=-62, threshold vth0=-52 for I spikes.
        'eLeakInh': -62 * mV,
        'vResetInh': -60 * mV,
        'vThreshInh': -52 * mV,
        'refrInh': 1 * ms,
        'membraneCapacitanceInh': 300 * pF,
        'gLeakInh': 15 * nS,

        'noiseSigma': 3 * mV,

        # Conductance step per presynaptic spike (sim.jl weightpars / synaptic table).
        'gEE': 2.86 * nS,
        'gEI': 48.7 * nS,
        'gIE': 1.27 * nS,
        'gII': 16.2 * nS,
        'weightCV': 0.1,

        'eRevExcSyn': 0 * mV,
        'eRevInhSyn': -75 * mV,
        # Spike increments dimensionless filter: x += g_syn / conductance_filter_ref (keep 1*nS so 2.86*nS → +2.86 like sim.jl).
        'conductance_filter_ref': 1 * nS,
        # ge_syn = (xd-xr)/tau_ms_diff * conductance_ge_scale. ~50–52*nS with full Julia weights hits a sharp
        # transition to ~kHz runaway; ~12*nS is subcritical — use spikeInputAmplitude to restore driven firing.
        'conductance_ge_scale': 11 * nS,

        'stdp_blocks': None,
        'tauu_vstdp': 10 * ms,
        'tauv_vstdp': 7 * ms,
        'taux_vstdp': 15 * ms,
        # vSTDP strengths (LTD: S/V, LTP: S/V^2). LTP multiplies by dt in ms in network.py to
        # match sim.jl's dt*altp*x*(v-thetaltp)*(v_lp-thetaltd) scaling.
        'altd_vstdp': 8e-10 * siemens / volt,
        'altp_vstdp': 1.4e-5 * siemens / (volt * volt),
        'thetaltd_vstdp': -70 * mV,
        'thetaltp_vstdp': -49 * mV,
        'stdp_delay': 0 * second,
        'g_min_EE': 0 * nS,
        'g_max_EE': 21.4 * nS,

        # iSTDP on I→E (sim.jl: tauy=20, eta=1, r0=0.003/ms → 3 Hz target, jeimin/jeimax for EI).
        'use_istdp': True,
        'istdp_tau_y': 20 * ms,
        'istdp_eta': 1 * nS,
        'istdp_r0': 3 * Hz,
        'g_min_EI': 48.7 * nS,
        'g_max_EI': 243 * nS,

        'use_homeostatic_norm': True,
        'homeostatic_norm_period': 20 * ms,
        'homeostatic_norm_beta': 1.0,

        'record_ee_w_stats': True,
        'w_stats_record_dt': 0.5 * second,
        # Manual trial-end rule for NS->US EE weights:
        # after each trial, NS neurons that fired in the US window potentiate NS->US;
        # NS neurons that did not fire in that window depress NS->US.
        'manual_ns_us_trial_learning': False,
        'manual_ns_us_trial_pot_delta': 0.3 * nS,
        'manual_ns_us_trial_dep_delta': 0.3 * nS,

        # sim.jl tauerise/tauedecay/tauirise/tauidecay (ms)
        'tauRiseExc': 1 * ms,
        'tauFallExc': 6 * ms,
        'tauRiseInh': 0.5 * ms,
        'tauFallInh': 2 * ms,
        'delayExc': 1 * ms,
        'delayInh': 0.5 * ms,

        'measure_mean_firing_rates': True,

        'n_record_voltage': 100,
        'save_checkpoint': True,
        'checkpoint_path': 'results/istdp_network_checkpoint.pkl',
        'load_checkpoint_path': None,
    }
    derive_trial_params(params)
    return params
