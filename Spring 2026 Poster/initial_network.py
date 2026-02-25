from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

from plotting import SimpleResults
from utils import (
    trial_duration,
    pulse_times_train,
    adjacency_indices_within,
    adjacency_indices_between,
    normal_weights,
    save_network_checkpoint,
)

rng_seed = 42
seed(rng_seed)
np.random.seed(rng_seed)
rng = np.random.default_rng(rng_seed)

# =============================
# Parameters
# =============================
params = {
    'dt': 0.1 * ms,

    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': True,

    # CS-US training (red = CS, blue = US; paper: 440 ms red @ 25 Hz, 80 ms blue @ 50 Hz)
    'nTrials': 50,
    'ISI': 360 * ms,              # time from CS onset to US onset (inter-stimulus interval)
    'propCS': 0.05,               # fraction of excitatory neurons selected for CS (red)
    'propUS': 0.05,               # fraction of excitatory neurons selected for US (blue)
    'interTrialInterval': 2 * second,
    'include_CS_only_trial': True,  # if True, add one extra trial with CS only (no US)
    'cs_only_every_n_trials': 10,  # if int (e.g. 5), every nth trial is CS only (no US) to probe training
    'CS_train_duration': 440 * ms,
    'CS_Hz': 25 * Hz,
    'US_train_duration': 80 * ms,
    'US_Hz': 50 * Hz,

    'spikeInputAmplitude': 0.98,  # current (nA) per CS/US pulse

    'nUnits': 2e3,
    'propInh': 0.20,
    'propConnect': 0.25,

    'eLeakExc': -65 * mV,
    'vResetExc': -58 * mV,
    'vThreshExc': -52 * mV,
    'betaAdaptExc': 10 * nA * ms,
    'refrExc': 2.5 * ms,
    'membraneCapacitanceExc': 200 * pF,
    'gLeakExc': 10 * nS,

    'eLeakInh': -65 * mV,
    'vResetInh': -58 * mV,
    'vThreshInh': -43 * mV,
    'betaAdaptInh': 1 * nA * ms,
    'refrInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,
    'gLeakInh': 8 * nS,

    'adaptTau': 500 * ms,

    'noiseSigma': 1 * mV,

    'jEE': 252 * pA,
    'jEI': 264 * pA,
    'jIE': 308 * pA,
    'jII': 282 * pA,
    'weightCV': 0.1,   # 10% std relative to mean

    # STDP (only EE when use_stdp is True)
    'use_stdp': True,
    'tau_stdp_pre': 20 * ms,
    'tau_stdp_post': 20 * ms,
    'A_plus_stdp': 5 * pA,   # LTP when pre before post
    'A_minus_stdp': 5 * pA,  # LTD when post before pre
    'w_min_EE': 0 * pA,
    'w_max_EE': 500 * pA,

    'tauRiseExc': 8 * ms,
    'tauFallExc': 23 * ms,
    'tauRiseInh': 1 * ms,
    'tauFallInh': 1 * ms,
    'delayExc': 1 * ms,
    'delayInh': 0.5 * ms,

    # Voltage recording: how many neurons per population to record (None = all)
    'n_record_voltage': 100,

    # Checkpoint: when True, save weights, params, and monitor data to a pickle file for later loading
    'save_checkpoint': False,
    'checkpoint_path': 'network_checkpoint.pkl',
}

# Derive duration from trial setup (no single duration param)
params['trialDuration'] = trial_duration(params) * second
params['trialPeriod'] = params['trialDuration'] + params['interTrialInterval']  # time from one trial start to next
n_trials_total = params['nTrials'] + (1 if params.get('include_CS_only_trial', False) else 0)
params['duration'] = (n_trials_total * float(params['trialPeriod'] / second) -
                      float(params['interTrialInterval'] / second) + 0.5) * second

defaultclock.dt = params['dt']

# =============================
# Neurons
# =============================
unitModel = '''
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

resetCode = '''
    v = vReset
    iAdapt += betaAdapt / tauAdapt 
    '''

threshCode = 'v >= vThresh'

params['nInh'] = int(params['propInh'] * params['nUnits'])
params['nExc'] = int(params['nUnits'] - params['nInh'])

unitsExc = NeuronGroup(
    N=params['nExc'],
    model=unitModel,
    method='euler',
    threshold=threshCode,
    reset=resetCode,
    refractory=params['refrExc'],
    clock=defaultclock,
)
unitsInh = NeuronGroup(
    N=params['nInh'],
    model=unitModel,
    method='euler',
    threshold=threshCode,
    reset=resetCode,
    refractory=params['refrInh'],
    clock=defaultclock,
)

mean_beta = params['betaAdaptExc']        # e.g. 0.02 * nA * ms
std_beta  = 0.2 * mean_beta

unitsExc.v = params['eLeakExc']
unitsExc.vReset = params['vResetExc']
unitsExc.vThresh = params['vThreshExc']
unitsExc.betaAdapt = mean_beta
# unitsExc.betaAdapt = np.random.normal(
#     loc=mean_beta / (amp*second),
#     scale=std_beta / (amp*second),
#     size=len(unitsExc)
# ) * amp * second
unitsExc.eLeak = params['eLeakExc']
unitsExc.Cm = params['membraneCapacitanceExc']
unitsExc.gl = params['gLeakExc']

unitsInh.v = params['eLeakInh']
unitsInh.vReset = params['vResetInh']
unitsInh.vThresh = params['vThreshInh']
unitsInh.betaAdapt = params['betaAdaptInh']
unitsInh.eLeak = params['eLeakInh']
unitsInh.Cm = params['membraneCapacitanceInh']
unitsInh.gl = params['gLeakInh']

# =============================
# CS-US training (recurring activation of CS and US neuron subsets)
# =============================
nCS = max(1, int(round(params['propCS'] * params['nExc'])))
nUS = max(1, int(round(params['propUS'] * params['nExc'])))
# Contiguous blocks: CS = [0, nCS), US = [nCS, nCS+nUS), other = [nCS+nUS, nExc)
nUS = min(nUS, params['nExc'] - nCS)  # keep CS+US within nExc
cs_neuron_inds = np.arange(0, nCS)
us_neuron_inds = np.arange(nCS, nCS + nUS)
params['cs_neuron_inds'] = cs_neuron_inds
params['us_neuron_inds'] = us_neuron_inds

trial_starts_s = np.array([tr * float(params['trialPeriod'] / second) for tr in range(n_trials_total)])

# CS on all trials
cs_times_s = pulse_times_train(
    trial_starts_s,
    float(params['CS_train_duration'] / second),
    float(params['CS_Hz'] / Hz),
)
# US only on paired trials: never on the optional extra CS-only trial; skip every nth if set
n_trials_paired = params['nTrials']  # extra trial at end (if any) is always CS-only
every_n = params.get('cs_only_every_n_trials')
if every_n is not None and every_n >= 1:
    # Trial i (0-indexed) is CS-only when (i+1) % every_n == 0
    trial_inds_US = [i for i in range(n_trials_paired) if (i + 1) % every_n != 0]
    trial_starts_for_US = trial_starts_s[trial_inds_US]
else:
    trial_starts_for_US = trial_starts_s[:n_trials_paired]
us_times_s = pulse_times_train(
    trial_starts_for_US + float(params['ISI'] / second),
    float(params['US_train_duration'] / second),
    float(params['US_Hz'] / Hz),
)

# Times when US would have started on CS-only trials (for marking on voltage plot)
ISI_s = float(params['ISI'] / second)
cs_only_trial_inds = []
if every_n is not None and every_n >= 1:
    cs_only_trial_inds.extend([i for i in range(n_trials_paired) if (i + 1) % every_n == 0])
if params.get('include_CS_only_trial', False) and n_trials_total > n_trials_paired:
    cs_only_trial_inds.append(n_trials_paired)
params['us_omit_times_s'] = np.array([trial_starts_s[i] + ISI_s for i in cs_only_trial_inds])
# Trial structure for PCA centroid trajectories: start times, duration, condition per trial
params['trial_starts_s'] = trial_starts_s
params['trial_duration_s'] = float(params['trialDuration'] / second)
cs_only_set = set(cs_only_trial_inds)
params['trial_conditions'] = np.array(['CS' if i in cs_only_set else 'US' for i in range(n_trials_total)])

# Expand to one spike per target neuron per pulse time
cs_indices_src = np.repeat(np.arange(nCS), len(cs_times_s))
cs_times_expanded = np.tile(cs_times_s, nCS)
us_indices_src = np.repeat(np.arange(nUS), len(us_times_s))
us_times_expanded = np.tile(us_times_s, nUS)

CS_group = SpikeGeneratorGroup(nCS, cs_indices_src, cs_times_expanded * second)
US_group = SpikeGeneratorGroup(nUS, us_indices_src, us_times_expanded * second)

syn_CS = Synapses(CS_group, unitsExc, on_pre='uExt_post += ' + str(params['spikeInputAmplitude']) + ' * nA')
syn_CS.connect(i=np.arange(nCS), j=cs_neuron_inds)
syn_US = Synapses(US_group, unitsExc, on_pre='uExt_post += ' + str(params['spikeInputAmplitude']) + ' * nA')
syn_US.connect(i=np.arange(nUS), j=us_neuron_inds)

# =============================
# Recurrent synapses
# =============================
tauRiseEOverMS = params['tauRiseExc'] / ms
tauRiseIOverMS = params['tauRiseInh'] / ms

# from E to E (with optional STDP)
if params.get('use_stdp', False):
    eqs_EE = '''
    jEE : amp
    dapre/dt = -apre / tau_stdp_pre : 1 (event-driven)
    dapost/dt = -apost / tau_stdp_post : 1 (event-driven)
    '''
    on_pre_EE = '''
    uE_post += jEE / tauRiseEOverMS
    apre += 1
    jEE = clip(jEE + A_plus_stdp * apost, w_min_EE, w_max_EE)
    '''
    on_post_EE = '''
    apost += 1
    jEE = clip(jEE - A_minus_stdp * apre, w_min_EE, w_max_EE)
    '''
    synapsesEE = Synapses(
        source=unitsExc,
        target=unitsExc,
        model=eqs_EE,
        on_pre=on_pre_EE,
        on_post=on_post_EE,
    )
    preInds, postInds = adjacency_indices_within(params['nExc'], params['propConnect'], rng)
    synapsesEE.connect(i=preInds, j=postInds)
    synapsesEE.jEE = normal_weights(params['jEE'], len(synapsesEE), params['weightCV'], rng)
    synapsesEE.apre = 0
    synapsesEE.apost = 0
    synapsesEE.namespace.update({
        'tau_stdp_pre': params['tau_stdp_pre'],
        'tau_stdp_post': params['tau_stdp_post'],
        'A_plus_stdp': params['A_plus_stdp'],
        'A_minus_stdp': params['A_minus_stdp'],
        'w_min_EE': params['w_min_EE'],
        'w_max_EE': params['w_max_EE'],
    })
else:
    synapsesEE = Synapses(
        model='jEE: amp',
        source=unitsExc,
        target=unitsExc,
        on_pre='uE_post += jEE / tauRiseEOverMS',
    )
    preInds, postInds = adjacency_indices_within(params['nExc'], params['propConnect'], rng)
    synapsesEE.connect(i=preInds, j=postInds)
    synapsesEE.jEE = normal_weights(params['jEE'], len(synapsesEE), params['weightCV'], rng)
paramsreEE = preInds
paramsosEE = postInds

# from E to I
synapsesEI = Synapses(
    model='jEI: amp',
    source=unitsInh,
    target=unitsExc,
    on_pre='uI_post += jEI / tauRiseIOverMS',
)
preInds, postInds = adjacency_indices_between(params['nInh'], params['nExc'], params['propConnect'], rng)
synapsesEI.connect(i=preInds, j=postInds)
paramsreEI = preInds
paramsosEI = postInds
synapsesEI.jEI = normal_weights(params['jEI'], len(synapsesEI), params['weightCV'], rng)

# from I to E
synapsesIE = Synapses(
    model='jIE: amp',
    source=unitsExc,
    target=unitsInh,
    on_pre='uE_post += jIE / tauRiseEOverMS',
)
preInds, postInds = adjacency_indices_between(params['nExc'], params['nInh'], params['propConnect'], rng)
synapsesIE.connect(i=preInds, j=postInds)
paramsreIE = preInds
paramsosIE = postInds
synapsesIE.jIE = normal_weights(params['jIE'], len(synapsesIE), params['weightCV'], rng)

# from I to I
synapsesII = Synapses(
    model='jII: amp',
    source=unitsInh,
    target=unitsInh,
    on_pre='uI_post += jII / tauRiseIOverMS',
)
preInds, postInds = adjacency_indices_within(params['nInh'], params['propConnect'], rng)
synapsesII.connect(i=preInds, j=postInds)
paramsreII = preInds
paramsosII = postInds
synapsesII.jII = normal_weights(params['jII'], len(synapsesII), params['weightCV'], rng)

synapsesEE.delay = ((rng.random(synapsesEE.delay.shape[0]) * params['delayExc'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt
synapsesEI.delay = ((rng.random(synapsesEI.delay.shape[0]) * params['delayInh'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt
synapsesIE.delay = ((rng.random(synapsesIE.delay.shape[0]) * params['delayExc'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt
synapsesII.delay = ((rng.random(synapsesII.delay.shape[0]) * params['delayInh'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt


def build_weight_matrix(syn_EE, syn_EI, syn_IE, syn_II, nExc, nInh):
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


# Store pre-simulation weight matrix in params
params['weight_matrix_pre'] = build_weight_matrix(
    synapsesEE, synapsesEI, synapsesIE, synapsesII,
    params['nExc'], params['nInh']
)

# =============================
# Monitors
# =============================
spikeMonExc = SpikeMonitor(unitsExc)
spikeMonInh = SpikeMonitor(unitsInh)

n_rec = params.get('n_record_voltage')
# Evenly spaced indices so we get a proportional spread across CS, US, other (not just 0..n_rec-1)
if n_rec is None:
    record_exc = True
    record_inh = True
else:
    n_re = min(int(n_rec), params['nExc'])
    n_ri = min(int(n_rec), params['nInh'])
    record_exc = np.linspace(0, params['nExc'] - 1, n_re, dtype=int)
    record_inh = np.linspace(0, params['nInh'] - 1, n_ri, dtype=int)
# Store which neuron indices are in each recorded row (for mapping in plots)
params['record_voltage_exc_inds'] = np.arange(params['nExc']) if record_exc is True else np.asarray(record_exc)
params['record_voltage_inh_inds'] = np.arange(params['nInh']) if record_inh is True else np.asarray(record_inh)
stateMonExc = StateMonitor(unitsExc, 'v', record=record_exc)
stateMonInh = StateMonitor(unitsInh, 'v', record=record_inh)

# =============================
# Running
# =============================

noiseSigma = params['noiseSigma']

tauRiseE = params['tauRiseExc']
tauFallE = params['tauFallExc']
tauRiseI = params['tauRiseInh']
tauFallI = params['tauFallInh']
tauAdapt = params['adaptTau']

run(params['duration'], report=params['reportType'], report_period=params['reportPeriod'], profile=params['doProfile'])

# Store post-simulation weight matrix in params
params['weight_matrix_post'] = build_weight_matrix(
    synapsesEE, synapsesEI, synapsesIE, synapsesII,
    params['nExc'], params['nInh']
)

# Save checkpoint (weights, params, spike/voltage data) to pickle if requested
if params.get('save_checkpoint', False):
    save_network_checkpoint(
        params['checkpoint_path'],
        params,
        spikeMonExc,
        spikeMonInh,
        stateMonExc,
        stateMonInh,
    )

# =============================
# Plots
# =============================
results = SimpleResults(
    spikeMonExc,
    spikeMonInh,
    stateMonExc,
    stateMonInh,
    params
)


# Figure 1: raster, firing rate, voltage (mean per group ± SEM)
fig1 = plt.figure(figsize=(8, 10))
ax_raster = fig1.add_subplot(3, 1, 1)
ax_rate = fig1.add_subplot(3, 1, 2)
ax_voltage = fig1.add_subplot(3, 1, 3)
results.plot_spike_raster(ax_raster)
results.plot_firing_rate(ax_rate)
results.plot_voltage_by_groups(ax_voltage)
fig1.tight_layout()

# Figure 2: PCA 3D (scaled up), within/between correlation, PCA variance
fig2 = plt.figure(figsize=(10, 12))
gs = fig2.add_gridspec(3, 1, height_ratios=[2, 1, 1])  # 3D plot gets 2x height
ax_pca = fig2.add_subplot(gs[0], projection='3d')
ax_corr = fig2.add_subplot(gs[1])
ax_var = fig2.add_subplot(gs[2])

results.plot_pca_3d_time_color(ax=ax_pca, use_upstate_only=True)
results.plot_within_between_correlations(ax=ax_corr)
results.plot_pca_variance(ax=ax_var, use_upstate_only=True)
fig2.tight_layout()

# Figure 3: pre- and post-simulation weight matrices (from results/params)
fig3 = results.plot_weight_change_blocks()
if fig3 is not None:
    fig3.tight_layout()

# Figure 4: PCA centroid trajectories (CS vs US) and centroid distance over time
fig4 = results.plot_pca_centroid_trajectories(bin_size=10*ms, n_components=3)
if fig4 is not None:
    fig4.tight_layout()

plt.show()
