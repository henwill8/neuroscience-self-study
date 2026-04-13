"""
Utility functions for the poster network and analysis.
"""
import pickle
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


def save_network_checkpoint(filepath, params):
    """
    Save a weights-only checkpoint: weight_matrix_post and dimensions (nExc, nInh).
    Params should already contain weight_matrix_post, nExc, nInh (e.g. after a run).
    """
    data = {
        'checkpoint_type': 'weights',
        'weight_matrix_post': np.asarray(params['weight_matrix_post'], dtype=float).copy(),
        'nExc': int(params['nExc']),
        'nInh': int(params['nInh']),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def cs_us_assembly_patterns(n_exc, cs_neuron_inds, us_neuron_inds):
    """
    Boolean membership (2, n_exc): row 0 = CS assembly, row 1 = US assembly.
    """
    P = np.zeros((2, int(n_exc)), dtype=bool)
    cs = np.atleast_1d(cs_neuron_inds).astype(np.int64)
    us = np.atleast_1d(us_neuron_inds).astype(np.int64)
    P[0, cs] = True
    P[1, us] = True
    return P


def compute_W_in_W_out_per_assembly(weights, i_ee, j_ee, patterns):
    """
    Mean EE weight within each assembly (W_in) vs mean weight of synapses **out** of the assembly (W_out).

    W_in: both pre and post in the assembly (needs ≥2 such synapses, else nan).
    W_out: pre in assembly, post **not** in assembly (outgoing from assembly only; not inbound).

    weights: 1d array (amp or float), same length as i_ee / j_ee.
    patterns: (n_patterns, n_exc) bool.
    Returns (W_in_per_assembly, W_out_per_assembly) as lists of length n_patterns (may contain nan).
    """
    weights = np.asarray(weights, dtype=np.float64).ravel()
    i_ee = np.asarray(i_ee, dtype=np.int64).ravel()
    j_ee = np.asarray(j_ee, dtype=np.int64).ravel()
    n_syn = len(weights)
    if len(i_ee) != n_syn or len(j_ee) != n_syn:
        raise ValueError("weights, i_ee, j_ee must have the same length")
    patterns = np.asarray(patterns, dtype=bool)
    n_patterns = patterns.shape[0]
    W_in_per_assembly = []
    W_out_per_assembly = []
    for k in range(n_patterns):
        in_k = np.where(patterns[k, :])[0]
        in_set = set(in_k.tolist())
        w_in_list = []
        w_out_list = []
        for idx in range(n_syn):
            pre, post = int(i_ee[idx]), int(j_ee[idx])
            w = float(weights[idx])
            pre_in = pre in in_set
            post_in = post in in_set
            if pre_in and post_in:
                w_in_list.append(w)
            elif pre_in and not post_in:
                w_out_list.append(w)
        W_in_per_assembly.append(np.mean(w_in_list) if len(w_in_list) >= 2 else np.nan)
        W_out_per_assembly.append(np.mean(w_out_list) if w_out_list else np.nan)
    return W_in_per_assembly, W_out_per_assembly


def load_weights_checkpoint(filepath):
    """
    Load a weights-only checkpoint; return (weight_matrix_post, nExc, nInh).
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if data.get('checkpoint_type') != 'weights':
        raise ValueError("Checkpoint must be weights-only; got checkpoint_type=%r" % data.get('checkpoint_type'))
    return (
        np.asarray(data['weight_matrix_post'], dtype=float),
        int(data['nExc']),
        int(data['nInh']),
    )