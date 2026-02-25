"""
Utility functions for the poster network and analysis.
"""
import pickle
import numpy as np
from brian2 import second, mV
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


def serialize_params(params):
    """Convert params dict to pickle-safe form (Brian2 Quantity -> float/ndarray)."""
    out = {}
    for k, v in params.items():
        if hasattr(v, 'dimensions'):
            u = get_unit(v.dimensions)
            arr = np.asarray(v / u)
            out[k] = float(arr.flat[0]) if arr.size == 1 else arr.copy()
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def save_network_checkpoint(filepath, params, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh):
    """
    Save weights, params, and monitor data to a pickle file for later loading.
    params should already contain weight_matrix_pre and weight_matrix_post (numpy arrays).
    """
    data = {
        'params': serialize_params(params),
        'spike_t_exc': np.asarray(spikeMonExc.t / second),
        'spike_i_exc': np.asarray(spikeMonExc.i),
        'spike_t_inh': np.asarray(spikeMonInh.t / second),
        'spike_i_inh': np.asarray(spikeMonInh.i),
        'state_v_exc': np.asarray(stateMonExc.v / mV),
        'state_v_inh': np.asarray(stateMonInh.v / mV),
        'state_dt': float(stateMonExc.clock.dt / second),
        'duration': float(params['duration'] / second),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_network_checkpoint(filepath):
    """Load a saved checkpoint dict from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)