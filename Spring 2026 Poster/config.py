"""
Default network parameters. Call get_default_params() to get a copy with derived fields set.
"""
import numpy as np
from brian2 import *
from utils import trial_duration


def derive_trial_params(params):
    """Update trialDuration, trialPeriod, and duration from other params. Call after overriding params."""
    params['trialDuration'] = trial_duration(params) * second
    params['trialPeriod'] = params['trialDuration'] + params['interTrialInterval']
    n_trials_total = params['nTrials'] + (1 if params.get('include_CS_only_trial', False) else 0)
    params['duration'] = (
        n_trials_total * float(params['trialPeriod'] / second)
        - float(params['interTrialInterval'] / second) + 0.5
    ) * second


def get_default_params():
    """Return the default params dict with trial duration and simulation length derived."""
    params = {
        'dt': 0.1 * ms,

        'reportType': 'stdout',
        'reportPeriod': 10 * second,
        'doProfile': True,

        # -------------------------------------------------------------------------
        # CS-US training (red = CS, blue = US; paper: 440 ms red @ 25 Hz, 80 ms blue @ 50 Hz)
        # -------------------------------------------------------------------------
        'nTrials': 50,
        'ISI': 360 * ms,              # time from CS onset to US onset (inter-stimulus interval)
        'propCS': 0.05,               # fraction of excitatory neurons selected for CS (red)
        'propUS': 0.05,               # fraction of excitatory neurons selected for US (blue)
        'interTrialInterval': 1 * second,
        'include_CS_only_trial': True,   # if True, add one extra trial with CS only (no US)
        'cs_only_every_n_trials': None,    # if int (e.g. 5), every nth trial is CS only (no US) to probe training
        'CS_train_duration': 440 * ms,
        'CS_Hz': 25 * Hz,
        'US_train_duration': 80 * ms,
        'US_Hz': 50 * Hz,

        'spikeInputAmplitude': 0.98,  # current (nA) per CS/US pulse
        # Small jitter breaks exact simultaneity across CS/US neurons so pair eSTDP sees
        # stable pre/post order within a timestep (otherwise LTD often wins for sync drive).
        'drive_spike_jitter': 0.1 * ms,

        # -------------------------------------------------------------------------
        # Network size and connectivity
        # -------------------------------------------------------------------------
        'nUnits': 2e3,
        'propInh': 0.20,
        'propConnect': 0.25,

        # -------------------------------------------------------------------------
        # Excitatory neuron (E) parameters
        # -------------------------------------------------------------------------
        'eLeakExc': -65 * mV,
        'vResetExc': -58 * mV,
        'vThreshExc': -52 * mV,
        'betaAdaptExc': 10 * nA * ms,
        'refrExc': 2.5 * ms,
        'membraneCapacitanceExc': 200 * pF,
        'gLeakExc': 10 * nS,

        # -------------------------------------------------------------------------
        # Inhibitory neuron (I) parameters
        # -------------------------------------------------------------------------
        'eLeakInh': -65 * mV,
        'vResetInh': -58 * mV,
        'vThreshInh': -43 * mV,
        'betaAdaptInh': 1 * nA * ms,
        'refrInh': 1 * ms,
        'membraneCapacitanceInh': 120 * pF,
        'gLeakInh': 8 * nS,

        'adaptTau': 500 * ms,
        'use_adaptation': True,   # if False, no adaptation current (iAdapt) in unit model

        'noiseSigma': 1 * mV,

        # -------------------------------------------------------------------------
        # Synaptic weights (recurrent)
        # -------------------------------------------------------------------------
        'jEE': 252 * pA,
        'jEI': 264 * pA,
        'jIE': 308 * pA,
        'jII': 282 * pA,
        'weightCV': 0.1,   # 10% std relative to mean

        # -------------------------------------------------------------------------
        # Pair-based additive eSTDP on EE (Hebbian, time-lag via exponential kernels).
        # r_stdp on E decays with tau_+ (pre trace); s_stdp decays with tau_- (post trace).
        # On post:  ΔjEE = +A_plus * r_stdp_pre   (pre-before-post, causal)
        # On pre:    ΔjEE = -A_minus * s_stdp_post (post-before-pre, anti-causal)
        # Amplitudes O(pA), same scale as jEE (~250 pA). stdp_blocks: per CS/US/NS pair toggles.
        # If tau_- >> tau_+, s_stdp stays high longer → presynaptic spikes see more LTD (often
        # depression-dominated under correlated drive). Defaults favor comparable / slightly
        # shorter tau_- than classical Bi&Poo-style long LTD tail.
        # -------------------------------------------------------------------------
        'use_stdp': True,
        'stdp_blocks': None,
        'e_stdp_tau_plus': 20 * ms,
        'e_stdp_tau_minus': 20 * ms,
        'e_stdp_A_plus': 5 * pA,
        'e_stdp_A_minus': 5 * pA,
        'w_min_EE': 0 * pA,
        'w_max_EE': 500 * pA,

        # -------------------------------------------------------------------------
        # Inhibitory STDP (iSTDP) for I->E synapses (sim.jl-style traces yE, yI)
        # -------------------------------------------------------------------------
        'use_istdp': True,
        'istdp_tau_y': 20 * ms,
        'r0': 3 * Hz,
        'Z': 1 * pA,
        'J_EI_min': 0 * pA,
        'J_EI_max': 500 * pA,

        # -------------------------------------------------------------------------
        # Homeostatic normalization of EE weights (row-sum toward initial totals).
        # -------------------------------------------------------------------------
        'use_homeostatic_norm': True,
        'homeostatic_norm_period': 20 * ms,

        # -------------------------------------------------------------------------
        # Synaptic dynamics and delays
        # -------------------------------------------------------------------------
        'tauRiseExc': 8 * ms,
        'tauFallExc': 23 * ms,
        'tauRiseInh': 1 * ms,
        'tauFallInh': 1 * ms,
        'delayExc': 1 * ms,
        'delayInh': 0.5 * ms,

        # -------------------------------------------------------------------------
        # EE assembly weight stats: W_in within assembly; W_out = pre in assembly, post outside
        # -------------------------------------------------------------------------
        'record_ee_w_stats': True,
        'w_stats_record_dt': 0.2 * second,

        # -------------------------------------------------------------------------
        # Analysis options
        # -------------------------------------------------------------------------
        'measure_ns_peak_firing': True,   # if True, compute peak avg firing time and variance for NS (non-stimulated) exc group

        # -------------------------------------------------------------------------
        # Recording and checkpoint
        # -------------------------------------------------------------------------

        # I dont think this one is working
        'n_record_voltage': 100,   # how many neurons per population to record (None = all)

        'save_checkpoint': True,  # when True, save weights to checkpoint_path
        'checkpoint_path': 'results/istdp_network_checkpoint.pkl',
        'load_checkpoint_path': None,  # if set, load weights from this file (params from get_default_params / overrides)
        # 'load_checkpoint_path': 'results/360_long_stdp_network_checkpoint.pkl',
    }
    derive_trial_params(params)
    return params
