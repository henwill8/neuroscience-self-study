from Tools.scripts.fixdiv import report
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
    'trialTime': 20 * second,
    'dt': 1 * ms,

    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': True,

    'nUnits': 2e3,
    'propInh': 0.2,
    'propConnect': 0.25,

    'eLeakExc': -65 * mV,
    'vResetExc': -58 * mV,
    'vThreshExc': -52 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 12 * nA * ms,
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

    # external kick
    'propKickedSpike': 0.05,
    'poissonLambda': 0.5 * Hz,
    'spikeInputAmplitude': 0.98
}

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
params['nIncInh'] = int(params['propConnect'] * params['nInh'])
params['nIncExc'] = int(params['propConnect'] * params['nExc'])

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

unitsExc.v = params['eLeakExc']
unitsExc.vReset = params['vResetExc']
unitsExc.vThresh = params['vThreshExc']
unitsExc.betaAdapt = params['betaAdaptExc']
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
# Poisson Kicks
# =============================
def poisson_single(rate, dt, duration):
    timeArray = np.arange(0, float(duration), float(dt))
    randArray = rng.random(*timeArray.shape)
    spikeBool = randArray < (rate * dt)

    times = timeArray[spikeBool]

    return times

kickTimes = poisson_single(params['poissonLambda'], params['dt'], params['trialTime'])

spikeUnits = int(np.round(params['propKickedSpike'] * params['nUnits']))

indices = []
times = []
for kickTime in kickTimes:
    indices.extend(list(range(spikeUnits)))
    times.extend([float(kickTime), ] * spikeUnits)

params['upPoissonTimes'] = kickTimes

Uppers = SpikeGeneratorGroup(spikeUnits, np.array(indices), np.array(times) * second)

feedforwardUpExc = Synapses(
    source=Uppers,
    target=unitsExc,
    on_pre='uExt_post += ' + str(params['spikeInputAmplitude']) + ' * nA'
)
feedforwardUpExc.connect('i==j')

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

from brian2.units import *

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

stateMonExc = StateMonitor(unitsExc, ['sE', 'sI', 'v'], record=0)
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

run(params['trialTime'], report=params['reportType'], report_period=params['reportPeriod'], profile=params['doProfile'])

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

mean_sE = np.mean(stateMonExc.sE[0] / pA)
mean_sI = np.mean(stateMonExc.sI[0] / pA)
print("Mean sE:", mean_sE)
print("Mean sI:", mean_sI)
print("Difference:", mean_sE - mean_sI)

deltaV = (mean_sE - mean_sI) * pA / params['gLeakExc']
print("Predicted shift (mV):", deltaV / mV)


fig, axs = plt.subplots(3, 1, figsize=(8, 8))

results.plot_spike_raster(axs[0])
results.plot_firing_rate(axs[1])
results.plot_voltage(axs[2], unitType='Exc', neuron_index=0)

plt.tight_layout()
plt.show()