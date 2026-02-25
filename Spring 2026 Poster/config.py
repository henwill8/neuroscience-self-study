"""
Default network parameters. Call get_default_params() to get a copy with derived fields set.
"""
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
        'interTrialInterval': 2 * second,
        'include_CS_only_trial': True,   # if True, add one extra trial with CS only (no US)
        'cs_only_every_n_trials': 10,    # if int (e.g. 5), every nth trial is CS only (no US) to probe training
        'CS_train_duration': 440 * ms,
        'CS_Hz': 25 * Hz,
        'US_train_duration': 80 * ms,
        'US_Hz': 50 * Hz,

        'spikeInputAmplitude': 0.98,  # current (nA) per CS/US pulse

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
        # STDP (only EE when use_stdp is True)
        # -------------------------------------------------------------------------
        'use_stdp': True,
        'tau_stdp_pre': 20 * ms,
        'tau_stdp_post': 20 * ms,
        'A_plus_stdp': 5 * pA,   # LTP when pre before post
        'A_minus_stdp': 5 * pA,  # LTD when post before pre
        'w_min_EE': 0 * pA,
        'w_max_EE': 500 * pA,

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
        # Recording and checkpoint
        # -------------------------------------------------------------------------
        'n_record_voltage': 100,   # how many neurons per population to record (None = all)
        'save_checkpoint': True,  # when True, save weights, params, and monitor data to pickle for later loading
        'checkpoint_path': 'results/network_checkpoint.pkl',
    }
    derive_trial_params(params)
    return params
