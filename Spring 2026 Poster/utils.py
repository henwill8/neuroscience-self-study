"""
Utility functions for the poster network and analysis.
"""
import numpy as np
from brian2 import second
from brian2.units import get_unit


def trial_duration(p):
    """Trial length in time (s) from CS and US params: max(CS_train, ISI + US_train)."""
    return max(float(p['CS_train_duration'] / second),
               float((p['ISI'] + p['US_train_duration']) / second))


def pulse_times_train(trial_starts_s, train_duration_s, freq_Hz):
    """For each trial start, generate pulse times at freq_Hz for train_duration_s."""
    if freq_Hz == 0:
        return np.array([])

    period_s = 1.0 / float(freq_Hz)
    times = []
    for t0 in trial_starts_s:
        t = 0.0
        while t < train_duration_s:
            times.append(t0 + t)
            t += period_s
    return np.array(times)


def adjacency_indices_within(nUnits, pConn, rng):
    """Random connectivity within a population (no autapses). Returns (preInds, postInds)."""
    bestNumberOfSynapses = int(np.round(pConn * nUnits ** 2))
    probabilityArray = np.full((nUnits, nUnits), 1 / (nUnits * (nUnits - 1)))
    probabilityArray[np.diag_indices_from(probabilityArray)] = 0
    if pConn > (nUnits - 1) / nUnits:
        bestNumberOfSynapses -= int(np.round(nUnits ** 2 * (pConn - (nUnits - 1) / nUnits)))
    indicesFlat = rng.choice(nUnits ** 2, bestNumberOfSynapses, replace=False, p=probabilityArray.ravel())
    preInds, postInds = np.unravel_index(indicesFlat, (nUnits, nUnits))
    return preInds, postInds


def adjacency_indices_between(nUnitsPre, nUnitsPost, pConn, rng):
    """Random connectivity between two populations. Returns (preInds, postInds)."""
    bestNumberOfSynapses = int(np.round(pConn * nUnitsPre * nUnitsPost))
    indicesFlat = rng.choice(nUnitsPre * nUnitsPost, bestNumberOfSynapses, replace=False)
    preInds, postInds = np.unravel_index(indicesFlat, (nUnitsPre, nUnitsPost))
    return preInds, postInds


def normal_weights(mean_current, n, weightCV, rng):
    """Sample n weights from normal(mean, weightCV*mean), clipped to >= 0, same dimensions as mean_current."""
    unit = get_unit(mean_current.dimensions)
    mean_value = float(mean_current / unit)
    std_value = weightCV * mean_value
    weights = rng.normal(loc=mean_value, scale=std_value, size=n)
    weights = np.clip(weights, 0, None)
    return weights * unit