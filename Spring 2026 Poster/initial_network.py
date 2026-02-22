from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

from plotting import SimpleResults

rng_seed = 42
seed(rng_seed)
np.random.seed(rng_seed)
rng = np.random.default_rng(rng_seed)

# =============================
# Parameters
# =============================
params = {
    'nTrials': 3,
    'dt': 0.1 * ms,

    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': True,

    # CS-US training (red = CS, blue = US; paper: 440 ms red @ 25 Hz, 80 ms blue @ 50 Hz)
    'ISI': 360 * ms,              # time from CS onset to US onset (inter-stimulus interval)
    'propCS': 0.05,               # fraction of excitatory neurons selected for CS (red)
    'propUS': 0.05,               # fraction of excitatory neurons selected for US (blue)
    'interTrialInterval': 2 * second,
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

    'tauRiseExc': 8 * ms,
    'tauFallExc': 23 * ms,
    'tauRiseInh': 1 * ms,
    'tauFallInh': 1 * ms,
    'delayExc': 1 * ms,
    'delayInh': 0.5 * ms,
}

# Derive duration from trial setup (no single duration param)
def _trial_duration(p):
    return max(float(p['CS_train_duration'] / second),
               float((p['ISI'] + p['US_train_duration']) / second))

params['trialDuration'] = _trial_duration(params) * second
params['trialPeriod'] = params['trialDuration'] + params['interTrialInterval']  # time from one trial start to next
params['duration'] = (params['nTrials'] * float(params['trialPeriod'] / second) -
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
cs_neuron_inds = rng.choice(params['nExc'], size=nCS, replace=False)
us_neuron_inds = rng.choice(params['nExc'], size=nUS, replace=False)
params['cs_neuron_inds'] = cs_neuron_inds
params['us_neuron_inds'] = us_neuron_inds

def _pulse_times_train(trial_starts_s, train_duration_s, freq_Hz, dt_s=0.0001):
    """For each trial start, generate pulse times at freq_Hz for train_duration_s."""
    period_s = 1.0 / float(freq_Hz)
    times = []
    for t0 in trial_starts_s:
        t = 0.0
        while t < train_duration_s:
            times.append(t0 + t)
            t += period_s
    return np.array(times)

trial_starts_s = np.array([tr * float(params['trialPeriod'] / second) for tr in range(params['nTrials'])])

cs_times_s = _pulse_times_train(
    trial_starts_s,
    float(params['CS_train_duration'] / second),
    float(params['CS_Hz'] / Hz),
)
us_times_s = _pulse_times_train(
    trial_starts_s + float(params['ISI'] / second),
    float(params['US_train_duration'] / second),
    float(params['US_Hz'] / Hz),
)

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
def adjacency_indices_within(nUnits, pConn):
    bestNumberOfSynapses = int(np.round(pConn * nUnits ** 2))

    probabilityArray = np.full((nUnits, nUnits), 1 / (nUnits * (nUnits - 1)))
    probabilityArray[np.diag_indices_from(probabilityArray)] = 0

    if pConn > (nUnits - 1) / nUnits:
        bestNumberOfSynapses -= int(np.round(nUnits ** 2 * (pConn - (nUnits - 1) / nUnits)))

    indicesFlat = rng.choice(nUnits ** 2, bestNumberOfSynapses, replace=False, p=probabilityArray.ravel())

    preInds, postInds = np.unravel_index(indicesFlat, (nUnits, nUnits))
    return preInds, postInds

def adjacency_indices_between(nUnitsPre, nUnitsPost, pConn):
    bestNumberOfSynapses = int(np.round(pConn * nUnitsPre * nUnitsPost))
    indicesFlat = rng.choice(nUnitsPre * nUnitsPost, bestNumberOfSynapses, replace=False)

    preInds, postInds = np.unravel_index(indicesFlat, (nUnitsPre, nUnitsPost))

    return preInds, postInds

def normal_weights(mean_current, syn):
    n = len(syn)

    # Extract the unit
    unit = get_unit(mean_current.dimensions)

    # Convert mean to dimensionless value
    mean_value = float(mean_current / unit)

    std_value = params['weightCV'] * mean_value

    # Sample dimensionless numbers
    weights = np.random.normal(
        loc=mean_value,
        scale=std_value,
        size=n
    )

    # Prevent negative weights
    weights = np.clip(weights, 0, None)

    # Re-attach unit
    return weights * unit


tauRiseEOverMS = params['tauRiseExc'] / ms
tauRiseIOverMS = params['tauRiseInh'] / ms

# from E to E
synapsesEE = Synapses(
    model='jEE: amp',
    source=unitsExc,
    target=unitsExc,
    on_pre='uE_post += jEE / tauRiseEOverMS',  # 'uE_post += 1 / tauRiseEOverMS'
)
preInds, postInds = adjacency_indices_within(params['nExc'], params['propConnect'])
synapsesEE.connect(i=preInds, j=postInds)
paramsreEE = preInds
paramsosEE = postInds
synapsesEE.jEE = normal_weights(params['jEE'], synapsesEE)

# from E to I
synapsesEI = Synapses(
    model='jEI: amp',
    source=unitsInh,
    target=unitsExc,
    on_pre='uI_post += jEI / tauRiseIOverMS',
)
preInds, postInds = adjacency_indices_between(params['nInh'], params['nExc'], params['propConnect'])
synapsesEI.connect(i=preInds, j=postInds)
paramsreEI = preInds
paramsosEI = postInds
synapsesEI.jEI = normal_weights(params['jEI'], synapsesEI)

# from I to E
synapsesIE = Synapses(
    model='jIE: amp',
    source=unitsExc,
    target=unitsInh,
    on_pre='uE_post += jIE / tauRiseEOverMS',
)
preInds, postInds = adjacency_indices_between(params['nExc'], params['nInh'], params['propConnect'])
synapsesIE.connect(i=preInds, j=postInds)
paramsreIE = preInds
paramsosIE = postInds
synapsesIE.jIE = normal_weights(params['jIE'], synapsesIE)

# from I to I
synapsesII = Synapses(
    model='jII: amp',
    source=unitsInh,
    target=unitsInh,
    on_pre='uI_post += jII / tauRiseIOverMS',
)
preInds, postInds = adjacency_indices_within(params['nInh'], params['propConnect'])
synapsesII.connect(i=preInds, j=postInds)
paramsreII = preInds
paramsosII = postInds
synapsesII.jII = normal_weights(params['jII'], synapsesII)

synapsesEE.delay = ((rng.random(synapsesEE.delay.shape[0]) * params['delayExc'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt
synapsesEI.delay = ((rng.random(synapsesEI.delay.shape[0]) * params['delayInh'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt
synapsesIE.delay = ((rng.random(synapsesIE.delay.shape[0]) * params['delayExc'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt
synapsesII.delay = ((rng.random(synapsesII.delay.shape[0]) * params['delayInh'] /
                     defaultclock.dt).astype(int) + 1) * defaultclock.dt


def log_synapse_stats(name, syn, varname):
    # Get weights
    w = getattr(syn, varname)[:]

    # Convert to numpy array and strip units
    w = np.array(w)
    if hasattr(w, 'unit'):
        w = w / w.unit

    print(f"\n{name} synapse weight stats ({varname}):")
    print(f"  Count: {w.size}")
    print(f"  Mean:  {np.mean(w):.6e}")
    print(f"  Std:   {np.std(w):.6e}")
    print(f"  Min:   {np.min(w):.6e}")
    print(f"  Max:   {np.max(w):.6e}")


def log_incoming_strength(name, syn, varname, n_post):
    # Get weights and indices
    weights = np.array(getattr(syn, varname)[:])
    post_inds = np.array(syn.j[:])  # postsynaptic indices

    # Strip units if present
    if hasattr(weights, 'unit'):
        weights = weights / weights.unit

    # Accumulate total incoming weight per postsyn neuron
    total_incoming = np.zeros(n_post)
    np.add.at(total_incoming, post_inds, weights)

    print(f"\n{name} incoming total synaptic strength per neuron:")
    print(f"  Mean: {np.mean(total_incoming):.6e}")
    print(f"  Std:  {np.std(total_incoming):.6e}")
    print(f"  Min:  {np.min(total_incoming):.6e}")
    print(f"  Max:  {np.max(total_incoming):.6e}")

    # Also compute in-degree
    indegree = np.zeros(n_post)
    np.add.at(indegree, post_inds, 1)

    print(f"  Mean in-degree: {np.mean(indegree):.2f}")
    print(f"  Std in-degree:  {np.std(indegree):.2f}")


log_synapse_stats("EE", synapsesEE, "jEE")
log_synapse_stats("EI", synapsesEI, "jEI")
log_synapse_stats("IE", synapsesIE, "jIE")
log_synapse_stats("II", synapsesII, "jII")

log_incoming_strength("EE", synapsesEE, "jEE", len(unitsExc))
log_incoming_strength("EI", synapsesEI, "jEI", len(unitsExc))
log_incoming_strength("IE", synapsesIE, "jIE", len(unitsInh))
log_incoming_strength("II", synapsesII, "jII", len(unitsInh))

# =============================
# Monitors
# =============================
spikeMonExc = SpikeMonitor(unitsExc)
spikeMonInh = SpikeMonitor(unitsInh)

stateMonExc = StateMonitor(unitsExc, 'v', record=True)
stateMonInh = StateMonitor(unitsInh, 'v', record=True)

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


fig = plt.figure(figsize=(8, 12))
ax_raster = fig.add_subplot(4, 1, 1)
ax_rate = fig.add_subplot(4, 1, 2)
ax_voltage = fig.add_subplot(4, 1, 3)
ax_pca = fig.add_subplot(4, 1, 4, projection='3d')

# Raster (excitatory CS | US | other, then inhibitory)
results.plot_spike_raster(ax_raster)

# Firing rate
results.plot_firing_rate(ax_rate)

# Voltage trace
results.plot_voltage(ax_voltage, unitType='Exc', neuron_index=0, mean=False)

# PCA: first 3 components with time as color
results.plot_pca_3d_time_color(ax=ax_pca, use_upstate_only=False)


plt.tight_layout()
plt.show()
