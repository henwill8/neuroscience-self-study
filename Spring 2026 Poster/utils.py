"""Helpers for the poster network."""
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
    Entries are conductances in siemens (float). Params should already contain
    weight_matrix_post, nExc, nInh (e.g. after a run).
    """
    data = {
        'checkpoint_type': 'weights',
        'weight_matrix_post': np.asarray(params['weight_matrix_post'], dtype=float).copy(),
        'nExc': int(params['nExc']),
        'nInh': int(params['nInh']),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def cs_us_ns_pattern_matrix(n_exc, cs_inds, us_inds):
    """
    Boolean (3, n_exc): rows are mutually exclusive membership for CS, US, and NS
    (non-stimulated excitatory), same grouping as the raster in plotting.SimpleResults.
    """
    cs = np.zeros(int(n_exc), dtype=bool)
    us = np.zeros(int(n_exc), dtype=bool)
    ci = np.asarray(cs_inds, dtype=np.int64)
    ui = np.asarray(us_inds, dtype=np.int64)
    cs[ci] = True
    us[ui] = True
    ns = ~(cs | us)
    return np.stack([cs, us, ns], axis=0)


def compute_W_in_W_out_per_assembly(g_EE, i_ee, j_ee, patterns):
    """W_in[k]: mean w>0 on j→i with both ends in k. W_out[k]: mean w>0 on j→i with pre in k, post not."""
    g_EE = np.asarray(g_EE, dtype=np.float64).ravel()
    i_ee = np.asarray(i_ee, dtype=np.int64).ravel()
    j_ee = np.asarray(j_ee, dtype=np.int64).ravel()
    n_patterns, _ = patterns.shape

    W_in_per_assembly = []
    W_out_per_assembly = []
    for k in range(n_patterns):
        if not np.any(patterns[k, :]):
            W_in_per_assembly.append(np.nan)
            W_out_per_assembly.append(np.nan)
            continue
        w_in_list = []
        w_out_list = []
        for idx in range(len(g_EE)):
            w = float(g_EE[idx])
            if w <= 0.0:
                continue
            pre = int(i_ee[idx])
            post = int(j_ee[idx])
            pre_k = bool(patterns[k, pre])
            post_k = bool(patterns[k, post])
            if pre_k and post_k:
                w_in_list.append(w)
            elif pre_k and not post_k:
                w_out_list.append(w)
        W_in_per_assembly.append(np.mean(w_in_list) if len(w_in_list) >= 2 else np.nan)
        W_out_per_assembly.append(np.mean(w_out_list) if w_out_list else np.nan)
    return W_in_per_assembly, W_out_per_assembly