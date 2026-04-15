import numpy as np
from scipy.special import gammaln
from sklearn.decomposition import PCA
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Black (zero weight) to yellow (max weight) for weight matrices
BLACK_YELLOW_CMAP = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'], N=256)


def _mean_per_neuron_binned_rate_hz(spike_times, spike_indices, neuron_ids, bins, bin_size_s):
    """
    Per time bin: average over neurons in ``neuron_ids`` of each neuron's rate (spikes in bin / bin_width).
    Neurons with no spikes in a bin contribute 0 Hz; silent neurons are included in the mean.
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=np.int64)
    neuron_ids = np.asarray(neuron_ids, dtype=np.int64).ravel()
    n_bins = len(bins) - 1
    if neuron_ids.size == 0:
        return np.zeros(n_bins, dtype=float)
    acc = np.zeros(n_bins, dtype=float)
    for nid in neuron_ids:
        tn = spike_times[spike_indices == nid]
        counts, _ = np.histogram(tn, bins)
        acc += counts.astype(float) / float(bin_size_s)
    return acc / float(neuron_ids.size)


def draw_cs_us_stimulus_spans(ax, p, zorder=0, legend_label=True):
    """
    Shade CS/US train windows from params cs_stim_intervals_s, us_stim_intervals_s (each row [t0, t1] s).
    Same as trial-level cs_stim_epoch_intervals_s / us_stim_epoch_intervals_s after scheduling.
    """
    labeled_cs = False
    labeled_us = False
    intervals_cs = p.get('cs_stim_intervals_s')
    intervals_us = p.get('us_stim_intervals_s')
    if intervals_cs is not None:
        arr = np.asarray(intervals_cs, dtype=float)
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(1, -1)
        for row in arr:
            if row.size < 2:
                continue
            t0, t1 = float(row[0]), float(row[1])
            if t1 < t0:
                t0, t1 = t1, t0
            show = legend_label and not labeled_cs
            ax.axvspan(
                t0, t1, facecolor='#3355aa', alpha=0.18, zorder=zorder, linewidth=0,
                label='CS stim' if show else None,
            )
            if show:
                labeled_cs = True
    if intervals_us is not None:
        arr = np.asarray(intervals_us, dtype=float)
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(1, -1)
        for row in arr:
            if row.size < 2:
                continue
            t0, t1 = float(row[0]), float(row[1])
            if t1 < t0:
                t0, t1 = t1, t0
            show = legend_label and not labeled_us
            ax.axvspan(
                t0, t1, facecolor='#aa5522', alpha=0.18, zorder=zorder, linewidth=0,
                label='US stim' if show else None,
            )
            if show:
                labeled_us = True


def draw_cs_us_stimulus_pulse_lines(ax, p, zorder=2, legend_label=True):
    """
    Vertical line at each nominal CS/US input pulse time (cs_stim_pulse_times_s, us_stim_pulse_times_s).
    Actual SpikeGenerator times are jittered per neuron when CS_input_jitter_std / US_input_jitter_std > 0
    (clamped to [0, epoch_start + train_duration]).
    """
    labeled_cs = False
    labeled_us = False
    t_cs = p.get('cs_stim_pulse_times_s')
    t_us = p.get('us_stim_pulse_times_s')
    if t_cs is not None and np.asarray(t_cs).size > 0:
        for tp in np.unique(np.asarray(t_cs, dtype=float)):
            if not np.isfinite(tp):
                continue
            show = legend_label and not labeled_cs
            ax.axvline(
                float(tp), color='#1a3a7a', ls='-', lw=0.75, alpha=0.65, zorder=zorder,
                label='CS pulse' if show else None,
            )
            if show:
                labeled_cs = True
    if t_us is not None and np.asarray(t_us).size > 0:
        for tp in np.unique(np.asarray(t_us, dtype=float)):
            if not np.isfinite(tp):
                continue
            show = legend_label and not labeled_us
            ax.axvline(
                float(tp), color='#7a3010', ls='-', lw=0.75, alpha=0.65, zorder=zorder,
                label='US pulse' if show else None,
            )
            if show:
                labeled_us = True


def _hmm_viterbi_poisson(counts, trans_mat, lambda_down, lambda_up):
    """
    Two-state (DOWN=0, UP=1) HMM with Poisson emissions. Viterbi decoding.
    counts: (n_bins,) integer spike counts per bin
    trans_mat: (2,2) P(s_next | s_curr), row i = current state, row j = next
    Returns: states (n_bins,) with values 0 or 1
    """
    n = len(counts)
    counts = np.asarray(counts, dtype=float)
    lambda_down = max(1e-10, float(lambda_down))
    lambda_up = max(1e-10, float(lambda_up))
    log_trans = np.log(trans_mat + 1e-20)

    def log_emit(s, y):
        lam = lambda_down if s == 0 else lambda_up
        return y * np.log(lam) - lam - gammaln(y + 1)

    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0, 0] = np.log(0.5) + log_emit(0, counts[0])
    delta[0, 1] = np.log(0.5) + log_emit(1, counts[0])

    for t in range(1, n):
        for s in range(2):
            scores = delta[t - 1, :] + log_trans[:, s] + log_emit(s, counts[t])
            psi[t, s] = np.argmax(scores)
            delta[t, s] = scores[psi[t, s]]

    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(delta[-1, :])
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ---------------------------------------------------------------------------
# Module-level computation functions (take results + optional params)
# ---------------------------------------------------------------------------

def compute_population_matrix(results, bin_size=5*ms, use_exc_only=True, subtract_mean=True):
    """Binned firing rate matrix (n_bins, n_neurons)."""
    bin_size_s = float(bin_size / second)
    bins = np.arange(0, results.duration, bin_size_s)
    n_neurons = results.p['nExc'] if use_exc_only else results.p['nUnits']
    X = np.zeros((len(bins) - 1, n_neurons))

    for i in range(results.p['nExc']):
        spikes = results.spikeMonExcT[results.spikeMonExcI == i]
        counts, _ = np.histogram(spikes, bins)
        X[:, i] = counts / bin_size_s  # Hz

    if not use_exc_only:
        for i in range(results.p['nInh']):
            spikes = results.spikeMonInhT[results.spikeMonInhI == i]
            counts, _ = np.histogram(spikes, bins)
            X[:, results.p['nExc'] + i] = counts / bin_size_s

    if subtract_mean:
        X = X - X.mean(axis=0)
    return X


def compute_pca_projection(results, bin_size=5*ms, use_exc_only=True, n_components=3):
    """Project binned population activity onto first n_components PCs. Returns (centers, proj)."""
    bin_size_s = float(bin_size / second)
    bins = np.arange(0, results.duration, bin_size_s)
    centers = bins[:-1] + bin_size_s / 2
    X = compute_population_matrix(results, bin_size, use_exc_only)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:n_components].T
    return centers, proj


def detect_upstates(results, bin_size=10*ms, use_exc_only=True,
                    p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0):
    """
    Label each time bin as UP or DOWN via two-state HMM on population spike count.
    Returns (centers, upstate_mask).
    """
    bin_size_s = float(bin_size / second)
    bins = np.arange(0, results.duration, bin_size_s)
    centers = bins[:-1] + bin_size_s / 2
    count, _ = np.histogram(results.spikeMonExcT, bins)
    count = np.asarray(count, dtype=float)

    mean_count = np.maximum(count.mean(), 1.0)
    r = rate_up_ratio / rate_down_ratio
    lambda_down = (2 * mean_count) / (1 + r)
    lambda_up = lambda_down * r

    p_switch = 1.0 - p_stay
    trans_mat = np.array([[p_stay, p_switch], [p_switch, p_stay]])
    states = _hmm_viterbi_poisson(count, trans_mat, lambda_down, lambda_up)
    upstate_mask = (states == 1)
    return centers, upstate_mask


def compute_within_between_correlations(results, bin_size=5*ms):
    """
    Mean correlation within (CS, US, NS) and between (CS-US, CS-NS, US-NS).
    Returns dict or None if CS/US not in params.
    Uses safe correlation (avoids divide-by-zero when a neuron has zero variance).
    """
    if 'cs_neuron_inds' not in results.p or 'us_neuron_inds' not in results.p:
        return None
    X = compute_population_matrix(results, bin_size, use_exc_only=True, subtract_mean=False)
    if X.shape[0] < 2 or X.shape[1] < 2:
        return {k: np.nan for k in ('within_CS', 'within_US', 'within_NS', 'between_CS_US', 'between_CS_NS', 'between_US_NS')}
    # Safe correlation: avoid divide-by-zero for constant (zero-variance) neurons
    C = np.cov(X.T)
    std = np.sqrt(np.maximum(np.diag(C), 0.0))
    std_safe = np.where(std > 1e-12, std, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = C / np.outer(std_safe, std_safe)
    np.fill_diagonal(R, 1.0)
    nExc = results.p['nExc']
    cs_set = set(results.p['cs_neuron_inds'])
    us_set = set(results.p['us_neuron_inds'])
    ns_inds = np.array([i for i in range(nExc) if i not in cs_set and i not in us_set])
    cs_inds = np.array(results.p['cs_neuron_inds'])
    us_inds = np.array(results.p['us_neuron_inds'])

    def mean_upper_triangle(R_sub):
        n = R_sub.shape[0]
        if n < 2:
            return np.nan
        triu = np.triu_indices(n, k=1)
        return np.nanmean(R_sub[triu])

    out = {}
    out['within_CS'] = mean_upper_triangle(R[np.ix_(cs_inds, cs_inds)]) if len(cs_inds) >= 2 else np.nan
    out['within_US'] = mean_upper_triangle(R[np.ix_(us_inds, us_inds)]) if len(us_inds) >= 2 else np.nan
    out['within_NS'] = mean_upper_triangle(R[np.ix_(ns_inds, ns_inds)]) if len(ns_inds) >= 2 else np.nan
    out['between_CS_US'] = np.nanmean(R[np.ix_(cs_inds, us_inds)]) if len(cs_inds) and len(us_inds) else np.nan
    out['between_CS_NS'] = np.nanmean(R[np.ix_(cs_inds, ns_inds)]) if len(cs_inds) and len(ns_inds) else np.nan
    out['between_US_NS'] = np.nanmean(R[np.ix_(us_inds, ns_inds)]) if len(us_inds) and len(ns_inds) else np.nan
    return out


def compute_pca_variance_explained(results, bin_size=5*ms, use_exc_only=True):
    """Variance explained by each PC (%). Full simulation; no upstate masking."""
    X = compute_population_matrix(results, bin_size, use_exc_only, subtract_mean=False)
    X = X - X.mean(axis=0)
    if X.shape[0] < 2:
        return np.full(3, np.nan)  # not enough data for variance explained
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = (S ** 2) / (S ** 2).sum()
    return var_explained * 100


def pca_condition_trajectories(data, conditions, n_components, condition_CS='CS', condition_US='US'):
    """
    Fit PCA on all trials combined, project each trial, compute condition centroids and their distance.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_timepoints, n_neurons)
        Trial-separated neural population data.
    conditions : array-like, shape (n_trials,)
        Condition label per trial (e.g. 'CS' or 'US', or 0/1). Must be comparable to condition_CS / condition_US.
    n_components : int
        Number of PCA components.
    condition_CS : str or scalar
        Label used for CS trials (default 'CS').
    condition_US : str or scalar
        Label used for US trials (default 'US').

    Returns
    -------
    projected_trials : ndarray, shape (n_trials, n_timepoints, n_components)
        Each trial projected into the shared PCA basis.
    centroid_CS : ndarray, shape (n_timepoints, n_components)
        Time-resolved centroid of CS trials in PCA space.
    centroid_US : ndarray, shape (n_timepoints, n_components)
        Time-resolved centroid of US trials in PCA space.
    centroid_distance : ndarray, shape (n_timepoints,)
        Euclidean distance between CS and US centroids at each timepoint.
    """
    data = np.asarray(data)
    conditions = np.asarray(conditions)
    n_trials, n_timepoints, n_neurons = data.shape

    # Fit PCA once on combined data: (samples, neurons)
    X_all = data.reshape(-1, n_neurons)
    pca = PCA(n_components=n_components)
    pca.fit(X_all)

    # Project each trial into the shared basis
    projected_trials = np.zeros((n_trials, n_timepoints, n_components))
    for t in range(n_trials):
        projected_trials[t] = pca.transform(data[t])  # (n_timepoints, n_components)

    # Condition masks
    is_CS = (conditions == condition_CS)
    is_US = (conditions == condition_US)
    n_CS = np.sum(is_CS)
    n_US = np.sum(is_US)

    # Time-resolved centroids (average across trials at each timepoint)
    centroid_CS = np.mean(projected_trials[is_CS], axis=0) if n_CS > 0 else np.full((n_timepoints, n_components), np.nan)
    centroid_US = np.mean(projected_trials[is_US], axis=0) if n_US > 0 else np.full((n_timepoints, n_components), np.nan)

    # Euclidean distance between centroids over time
    centroid_distance = np.linalg.norm(centroid_CS - centroid_US, axis=1)

    return projected_trials, centroid_CS, centroid_US, centroid_distance


def compute_trial_binned_data(results, bin_size=5*ms, use_exc_only=True):
    """
    Build trial-separated binned firing rate matrix from results.
    Requires params: trial_starts_s, trial_duration_s, trial_conditions.

    Returns
    -------
    data : ndarray, shape (n_trials, n_timepoints, n_neurons), or None if trial info missing
    conditions : ndarray, shape (n_trials,), or None
    """
    p = results.p
    if 'trial_starts_s' not in p or 'trial_duration_s' not in p or 'trial_conditions' not in p:
        return None, None
    trial_starts_s = np.asarray(p['trial_starts_s'])
    trial_duration_s = float(p['trial_duration_s'])
    conditions = np.asarray(p['trial_conditions'])
    bin_size_s = float(bin_size / second)
    n_trials = len(trial_starts_s)
    n_timepoints = int(round(trial_duration_s / bin_size_s))
    if n_timepoints < 1:
        return None, None
    n_neurons = p['nExc'] if use_exc_only else p['nUnits']
    data = np.zeros((n_trials, n_timepoints, n_neurons))
    for tr in range(n_trials):
        t0 = trial_starts_s[tr]
        t1 = t0 + trial_duration_s
        bins = np.linspace(t0, t1, n_timepoints + 1)
        for i in range(p['nExc']):
            spikes = results.spikeMonExcT[results.spikeMonExcI == i]
            counts, _ = np.histogram(spikes, bins)
            data[tr, :, i] = counts[:n_timepoints] / bin_size_s
        if not use_exc_only:
            for i in range(p['nInh']):
                spikes = results.spikeMonInhT[results.spikeMonInhI == i]
                counts, _ = np.histogram(spikes, bins)
                data[tr, :, p['nExc'] + i] = counts[:n_timepoints] / bin_size_s
    return data, conditions


def compute_mean_firing_rates(results):
    """
    Population-mean firing rates (Hz) for excitatory and inhibitory units over the full run:
    total_spikes / (n_neurons * duration).

    Returns
    -------
    dict with keys mean_rate_E_Hz, mean_rate_I_Hz, duration_s, n_spikes_E, n_spikes_I, nExc, nInh,
    or None if duration is non-positive or population sizes missing.
    """
    p = results.p
    T = float(p['duration'] / second)
    if T <= 0:
        return None
    n_e = int(p['nExc'])
    n_i = int(p['nInh'])
    if n_e < 1 or n_i < 1:
        return None
    n_spike_e = int(len(results.spikeMonExcT))
    n_spike_i = int(len(results.spikeMonInhT))
    return {
        'mean_rate_E_Hz': n_spike_e / (n_e * T),
        'mean_rate_I_Hz': n_spike_i / (n_i * T),
        'duration_s': T,
        'n_spikes_E': n_spike_e,
        'n_spikes_I': n_spike_i,
        'nExc': n_e,
        'nInh': n_i,
    }


def compute_ns_trial_trajectories(results, bin_size=5*ms, n_components=3):
    """
    Bin NS (non-stimulated) excitatory firing over the entire simulation, fit PCA on all bins,
    and return one trajectory in PC space (not reset per trial).

    Requires params: duration (or results.duration), cs_neuron_inds, us_neuron_inds, nExc.

    Returns
    -------
    projected : ndarray, shape (n_timepoints, n_components)
        NS population trajectory in PCA space for the full run.
    time_s : ndarray, shape (n_timepoints,)
        Simulation time (s) at bin centers.
    conditions : None
        Reserved; per-trial conditions are not used for full-run PCA.
    pca : sklearn PCA
        Fitted on (n_timepoints, n_ns_neurons) NS binned rates.
    ns_inds : ndarray
        Indices of NS neurons.
    or (None, None, None, None, None) if CS/US info missing, no NS neurons, or too few bins.
    """
    p = results.p
    if 'cs_neuron_inds' not in p or 'us_neuron_inds' not in p:
        return None, None, None, None, None
    cs_set = set(np.atleast_1d(p['cs_neuron_inds']))
    us_set = set(np.atleast_1d(p['us_neuron_inds']))
    n_exc = int(p['nExc'])
    ns_inds = np.array([i for i in range(n_exc) if i not in cs_set and i not in us_set])
    if len(ns_inds) == 0:
        return None, None, None, None, None

    T = float(getattr(results, 'duration', float(p['duration'] / second)))
    bin_size_s = float(bin_size / second)
    if T <= 0 or bin_size_s <= 0:
        return None, None, None, None, None
    n_timepoints = int(round(T / bin_size_s))
    if n_timepoints < 1:
        return None, None, None, None, None
    bins = np.linspace(0.0, T, n_timepoints + 1)
    n_ns = len(ns_inds)
    ns_data = np.zeros((n_timepoints, n_ns))
    for j, i in enumerate(ns_inds):
        spikes = results.spikeMonExcT[results.spikeMonExcI == i]
        counts, _ = np.histogram(spikes, bins)
        ns_data[:, j] = counts / bin_size_s

    n_components = min(int(n_components), n_ns, n_timepoints - 1)
    if n_components < 1:
        return None, None, None, None, None
    pca = PCA(n_components=n_components)
    pca.fit(ns_data)
    projected = pca.transform(ns_data)
    time_s = (bins[:-1] + bins[1:]) / 2.0
    return projected, time_s, None, pca, ns_inds


def _binary_auc_from_scores(y_true, scores):
    """
    ROC-AUC for binary labels {0,1} using rank statistic.
    Returns np.nan when undefined (single class or empty).
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if y_true.size == 0 or scores.size != y_true.size:
        return np.nan
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def compute_ns_time_resolved_decodability(results, bin_size=10*ms, n_splits=5):
    """
    Time-resolved NS decodability between CS and US trials using a nearest-centroid decoder.
    Returns (time_s, accuracy), or (None, None) if trial labels/NS group are unavailable.
    """
    p = results.p
    if 'cs_neuron_inds' not in p or 'us_neuron_inds' not in p:
        return None, None
    data, conditions = compute_trial_binned_data(results, bin_size=bin_size, use_exc_only=True)
    if data is None or conditions is None or data.size == 0:
        return None, None
    cond = np.asarray(conditions)
    mask = (cond == 'CS') | (cond == 'US')
    if np.sum(mask) < 4:
        return None, None
    data = data[mask]
    cond = cond[mask]
    y = (cond == 'US').astype(np.int64)

    cs_set = set(np.atleast_1d(p['cs_neuron_inds']))
    us_set = set(np.atleast_1d(p['us_neuron_inds']))
    n_exc = int(p['nExc'])
    ns_inds = np.array([i for i in range(n_exc) if i not in cs_set and i not in us_set], dtype=np.int64)
    if ns_inds.size == 0:
        return None, None
    ns_data = data[:, :, ns_inds]  # (n_trials, n_timepoints, n_ns)
    n_trials, n_timepoints, _ = ns_data.shape
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n_folds = int(min(max(2, n_splits), len(idx0), len(idx1)))
    if n_folds < 2:
        return None, None

    rng = np.random.default_rng(0)
    idx0 = idx0[rng.permutation(len(idx0))]
    idx1 = idx1[rng.permutation(len(idx1))]
    folds0 = np.array_split(idx0, n_folds)
    folds1 = np.array_split(idx1, n_folds)

    acc = np.full(n_timepoints, np.nan, dtype=float)
    for t in range(n_timepoints):
        Xt = ns_data[:, t, :]
        fold_scores = []
        for k in range(n_folds):
            test_idx = np.concatenate([folds0[k], folds1[k]])
            train_mask = np.ones(n_trials, dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]
            if train_idx.size < 2 or np.sum(y[train_idx] == 0) == 0 or np.sum(y[train_idx] == 1) == 0:
                continue
            c0 = Xt[train_idx][y[train_idx] == 0].mean(axis=0)
            c1 = Xt[train_idx][y[train_idx] == 1].mean(axis=0)
            d0 = np.sum((Xt[test_idx] - c0) ** 2, axis=1)
            d1 = np.sum((Xt[test_idx] - c1) ** 2, axis=1)
            y_hat = (d1 < d0).astype(np.int64)
            fold_scores.append(np.mean(y_hat == y[test_idx]))
        if len(fold_scores) > 0:
            acc[t] = float(np.mean(fold_scores))

    bin_size_s = float(bin_size / second)
    time_s = (np.arange(n_timepoints) + 0.5) * bin_size_s
    return time_s, acc


def compute_reservoir_memory_curve(results, bin_size=10*ms, max_lag_bins=40):
    """
    Memory proxy: how well current NS state predicts CS pulse count from k bins in the past.
    Returns (lags_s, r2_by_lag), or (None, None) if required signals are unavailable.
    """
    p = results.p
    if 'cs_neuron_inds' not in p or 'us_neuron_inds' not in p:
        return None, None
    cs_set = set(np.atleast_1d(p['cs_neuron_inds']))
    us_set = set(np.atleast_1d(p['us_neuron_inds']))
    n_exc = int(p['nExc'])
    ns_inds = np.array([i for i in range(n_exc) if i not in cs_set and i not in us_set], dtype=np.int64)
    if ns_inds.size == 0:
        return None, None

    bin_size_s = float(bin_size / second)
    T = float(getattr(results, 'duration', float(p['duration'] / second)))
    if T <= 0.0 or bin_size_s <= 0.0:
        return None, None
    n_bins = int(round(T / bin_size_s))
    if n_bins < 10:
        return None, None
    bins = np.linspace(0.0, T, n_bins + 1)

    X = np.zeros((n_bins, ns_inds.size), dtype=float)
    for j, i in enumerate(ns_inds):
        spikes = results.spikeMonExcT[results.spikeMonExcI == i]
        counts, _ = np.histogram(spikes, bins)
        X[:, j] = counts / bin_size_s
    X = X - X.mean(axis=0, keepdims=True)

    cs_pulses = np.asarray(p.get('cs_stim_pulse_times_s', []), dtype=float)
    if cs_pulses.size == 0:
        return None, None
    u, _ = np.histogram(cs_pulses, bins)
    u = u.astype(float)

    max_lag = int(min(max_lag_bins, n_bins // 2))
    if max_lag < 1:
        return None, None
    lags = np.arange(1, max_lag + 1, dtype=int)
    r2 = np.full(lags.size, np.nan, dtype=float)

    ridge = 1e-3
    for ii, lag in enumerate(lags):
        X_lag = X[lag:, :]
        y_lag = u[:-lag]
        n = X_lag.shape[0]
        n_train = int(np.floor(0.7 * n))
        if n_train < 5 or (n - n_train) < 3:
            continue
        Xtr, Xte = X_lag[:n_train], X_lag[n_train:]
        ytr, yte = y_lag[:n_train], y_lag[n_train:]

        Xtr_aug = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
        Xte_aug = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
        I = np.eye(Xtr_aug.shape[1], dtype=float)
        I[-1, -1] = 0.0  # do not regularize bias
        w = np.linalg.solve(Xtr_aug.T @ Xtr_aug + ridge * I, Xtr_aug.T @ ytr)
        yhat = Xte_aug @ w
        ss_res = np.sum((yte - yhat) ** 2)
        ss_tot = np.sum((yte - np.mean(yte)) ** 2)
        if ss_tot > 0:
            r2[ii] = 1.0 - ss_res / ss_tot

    return lags * bin_size_s, r2


def compute_us_readout_metrics(results):
    """
    Trial-level US readout metrics from US-neuron firing in the scheduled US train window.
    Returns dict or None if trial/US metadata is unavailable.
    """
    p = results.p
    needed = ('trial_starts_s', 'trial_conditions', 'us_neuron_inds', 'ISI', 'US_train_duration')
    if any(k not in p for k in needed):
        return None
    us_inds = np.atleast_1d(p['us_neuron_inds']).astype(np.int64)
    if us_inds.size == 0:
        return None
    trial_starts = np.asarray(p['trial_starts_s'], dtype=float)
    conditions = np.asarray(p['trial_conditions'])
    if trial_starts.size == 0 or conditions.size != trial_starts.size:
        return None

    us_start_offset = float(p['ISI'] / second)
    us_end_offset = us_start_offset + float(p['US_train_duration'] / second)
    if us_end_offset <= us_start_offset:
        return None
    window_s = us_end_offset - us_start_offset

    t_exc = np.asarray(results.spikeMonExcT, dtype=float)
    i_exc = np.asarray(results.spikeMonExcI, dtype=np.int64)
    in_us_group = np.isin(i_exc, us_inds)
    t_us_group = t_exc[in_us_group]

    rates = np.zeros(trial_starts.size, dtype=float)
    for tr, t0 in enumerate(trial_starts):
        a = t0 + us_start_offset
        b = t0 + us_end_offset
        n_spk = np.sum((t_us_group >= a) & (t_us_group < b))
        rates[tr] = n_spk / (us_inds.size * window_s)

    valid = (conditions == 'CS') | (conditions == 'US')
    if np.sum(valid) < 2:
        return None
    y = (conditions[valid] == 'US').astype(np.int64)
    s = rates[valid]
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return None

    m1 = float(np.mean(s[y == 1]))
    m0 = float(np.mean(s[y == 0]))
    v1 = float(np.var(s[y == 1]))
    v0 = float(np.var(s[y == 0]))
    dprime = (m1 - m0) / np.sqrt(0.5 * (v1 + v0) + 1e-12)
    threshold = 0.5 * (m1 + m0)
    acc = float(np.mean((s >= threshold).astype(np.int64) == y))
    auc = float(_binary_auc_from_scores(y, s))

    return {
        'trial_conditions': conditions,
        'trial_rates_Hz': rates,
        'valid_mask': valid,
        'auc': auc,
        'dprime': float(dprime),
        'threshold_acc': acc,
        'us_window_s': (us_start_offset, us_end_offset),
    }


def compute_block_weight_change(W_pre, W_post, groups, group_names=None):
    """
    Block-averaged weight change: for each (pre, post) connection block, compute mean and SEM
    of percentage change (W_post - W_pre) / W_pre * 100 over synapses with W_pre > 0.
    Weight matrix convention: W[post, pre] = weight from pre to post.

    groups: 1d array of length N, group index per neuron (0=CS, 1=US, 2=NS, etc.)
    group_names: optional list of names for labels (e.g. ['CS', 'US', 'NS'])
    Returns: labels as pre→post (e.g. ['CS→CS', 'CS→US', ...]), means (%), sems (%).
    """
    W_pre = np.asarray(W_pre)
    W_post = np.asarray(W_post)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if group_names is None:
        group_names = [str(g) for g in unique]
    else:
        group_names = [group_names[i] for i in range(len(unique))]
    labels = []
    means = []
    sems = []
    for i_post, g_post in enumerate(unique):
        for i_pre, g_pre in enumerate(unique):
            mask_post = (groups == g_post)
            mask_pre = (groups == g_pre)
            w_pre_b = W_pre[np.ix_(mask_post, mask_pre)].ravel()
            w_post_b = W_post[np.ix_(mask_post, mask_pre)].ravel()
            valid = w_pre_b > 0
            if np.sum(valid) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                w_pre_v = w_pre_b[valid]
                w_post_v = w_post_b[valid]
                pct = (w_post_v - w_pre_v) / w_pre_v * 100.0
                means.append(float(np.mean(pct)))
                n = len(pct)
                sems.append(float(np.std(pct) / np.sqrt(n)) if n > 1 else 0.0)
            labels.append(f"{group_names[i_pre]}→{group_names[i_post]}")
    return labels, np.array(means), np.array(sems)


# ---------------------------------------------------------------------------
# SimpleResults: data container + plotting only
# ---------------------------------------------------------------------------

class SimpleResults:
    """Holds spike/voltage data and params. Plotting methods only; computation is in module-level functions."""

    def __init__(self, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh, params):
        self.p = params
        self.spikeMonExcT = spikeMonExc.t / second
        self.spikeMonExcI = spikeMonExc.i
        self.spikeMonInhT = spikeMonInh.t / second
        self.spikeMonInhI = spikeMonInh.i
        self.stateMonExcV = np.asarray(stateMonExc.v / mV)
        self.stateMonInhV = np.asarray(stateMonInh.v / mV)
        self.stateDT = float(stateMonExc.clock.dt / second)
        self.duration = float(params['duration'] / second)
        # Time axis: match length to recorded voltage (StateMonitor is neurons x time)
        n_times = self.stateMonExcV.shape[1]
        self.stateMonT = np.arange(n_times, dtype=float) * self.stateDT
        # Optional: L&D-style W_in / W_out time series for CS, US, NS (from Network.run)
        self.w_stats_t = np.asarray(params.get('w_stats_t', []), dtype=float)
        self.w_in_CS_US_NS = np.asarray(params.get('w_in_CS_US_NS', np.zeros((0, 3))), dtype=float)
        self.w_out_CS_US_NS = np.asarray(params.get('w_out_CS_US_NS', np.zeros((0, 3))), dtype=float)

    def plot_spike_raster(self, ax):
        nExc = self.p['nExc']
        nInh = self.p['nInh']
        draw_cs_us_stimulus_spans(ax, self.p, zorder=0, legend_label=True)
        draw_cs_us_stimulus_pulse_lines(ax, self.p, zorder=2, legend_label=True)
        if 'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p:
            cs_set = set(self.p['cs_neuron_inds'])
            us_set = set(self.p['us_neuron_inds'])
            cs_sorted = np.sort(self.p['cs_neuron_inds'])
            us_sorted = np.sort(self.p['us_neuron_inds'])
            ns_sorted = np.sort([i for i in range(nExc) if i not in cs_set and i not in us_set])
            order = np.concatenate([cs_sorted, us_sorted, ns_sorted])
            neuron_to_display = {n: i for i, n in enumerate(order)}
            nCS, nUS = len(cs_sorted), len(us_sorted)
            t_exc = np.asarray(self.spikeMonExcT)
            i_exc = np.asarray(self.spikeMonExcI)
            y_exc = np.array([neuron_to_display[i] for i in i_exc])
            ax.scatter(t_exc, y_exc, c='blue', s=1, marker='.', linewidths=0, zorder=3)
            ax.axhline(nCS - 0.5, color='k', lw=0.5, linestyle='-')
            ax.axhline(nCS + nUS - 0.5, color='k', lw=0.5, linestyle='-')
            ax.scatter(self.spikeMonInhT, nExc + self.spikeMonInhI, s=1, c='red', marker='.', linewidths=0)
            ax.axhline(nExc - 0.5, color='k', lw=0.5, linestyle='-')
            ax.set_ylim(-0.5, self.p['nUnits'] - 0.5)
            ax.set_ylabel("Neuron (CS | US | NS | inh)")
        else:
            t_exc = np.asarray(self.spikeMonExcT)
            i_exc = np.asarray(self.spikeMonExcI)
            ax.scatter(t_exc, i_exc, s=1, c='cyan', marker='.', zorder=3)
            ax.scatter(self.spikeMonInhT, nExc + self.spikeMonInhI, s=1, c='red', marker='.', linewidths=0)
            ax.set_ylim(-0.5, self.p['nUnits'] - 0.5)
            ax.set_ylabel("Neuron index")
        ax.set_xlim(0, self.duration)
        ax.set_xlabel("Time (s)")
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    def plot_firing_rate(self, ax, bin_size=5*ms, show_upstate=True,
                         upstate_bin_size=10*ms, p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0):
        """Per time bin: mean over neurons in each group of that neuron's binned rate (Hz)."""
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        nExc = self.p['nExc']
        nInh = self.p['nInh']
        draw_cs_us_stimulus_spans(ax, self.p, zorder=0, legend_label=True)
        drew_upstate_span = False
        if show_upstate:
            _, upstate_mask = detect_upstates(self, bin_size=upstate_bin_size, use_exc_only=True,
                                              p_stay=p_stay, rate_up_ratio=rate_up_ratio, rate_down_ratio=rate_down_ratio)
            first_span = True
            upstate_bin_size_s = float(upstate_bin_size / second)
            i = 0
            while i < len(upstate_mask):
                if not upstate_mask[i]:
                    i += 1
                    continue
                start_i = i
                while i < len(upstate_mask) and upstate_mask[i]:
                    i += 1
                t_start = (start_i) * upstate_bin_size_s
                t_end = (i) * upstate_bin_size_s
                ax.axvspan(t_start, t_end, alpha=0.25, color='C1',
                           label='Upstate (HMM)' if first_span else None)
                first_span = False
                drew_upstate_span = True

        if 'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p:
            cs_set = set(self.p['cs_neuron_inds'])
            us_set = set(self.p['us_neuron_inds'])
            nCS, nUS = len(cs_set), len(us_set)
            nNS = nExc - nCS - nUS
            # One matrix (n_bins, nExc): each column is one neuron's binned rate (Hz).
            rates_exc = compute_population_matrix(self, bin_size, use_exc_only=True, subtract_mean=False)
            if nCS > 0:
                cs_ids = np.sort(np.asarray(self.p['cs_neuron_inds'], dtype=np.int64).ravel())
                FR_CS = rates_exc[:, cs_ids].mean(axis=1)
                ax.plot(centers, FR_CS, color='C3', alpha=0.8, label='CS')
            if nUS > 0:
                us_ids = np.sort(np.asarray(self.p['us_neuron_inds'], dtype=np.int64).ravel())
                FR_US = rates_exc[:, us_ids].mean(axis=1)
                ax.plot(centers, FR_US, color='C0', alpha=0.8, label='US')
            if nNS > 0:
                ns_ids = np.sort(np.array([i for i in range(nExc) if i not in cs_set and i not in us_set], dtype=np.int64))
                FR_ns = rates_exc[:, ns_ids].mean(axis=1)
                ax.plot(centers, FR_ns, color='0.6', alpha=0.6, label='NS')
            t_inh = np.asarray(self.spikeMonInhT, dtype=float)
            i_inh = np.asarray(self.spikeMonInhI, dtype=np.int64)
            inh_ids = np.arange(nInh, dtype=np.int64)
            FRInh = _mean_per_neuron_binned_rate_hz(t_inh, i_inh, inh_ids, bins, bin_size_s)
            ax.plot(centers, FRInh, color='red', alpha=0.6, label='inh')
        else:
            rates_exc = compute_population_matrix(self, bin_size, use_exc_only=True, subtract_mean=False)
            FRExc = rates_exc.mean(axis=1)
            t_inh = np.asarray(self.spikeMonInhT, dtype=float)
            i_inh = np.asarray(self.spikeMonInhI, dtype=np.int64)
            inh_ids = np.arange(nInh, dtype=np.int64)
            FRInh = _mean_per_neuron_binned_rate_hz(t_inh, i_inh, inh_ids, bins, bin_size_s)
            ax.plot(centers, FRExc, color='cyan', alpha=0.6)
            ax.plot(centers, FRInh, color='red', alpha=0.6)

        draw_cs_us_stimulus_pulse_lines(ax, self.p, zorder=3, legend_label=True)

        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        has_stim_spans = False
        for key in (
            'cs_stim_intervals_s', 'us_stim_intervals_s',
            'cs_stim_pulse_times_s', 'us_stim_pulse_times_s',
        ):
            v = self.p.get(key)
            if v is not None and np.asarray(v).size > 0:
                has_stim_spans = True
                break
        if (show_upstate and drew_upstate_span) or (
            'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p
        ) or has_stim_spans:
            ax.legend(loc='upper right', fontsize=7)

    def plot_voltage(self, ax, unitType='Exc', neuron_index=0, mean=False,
                    spike_peak_mV=0.0):
        if unitType == 'Exc':
            V = np.asarray(self.stateMonExcV)
            record_inds = self.p.get('record_voltage_exc_inds')
            record_inds = np.arange(V.shape[0]) if record_inds is None else np.atleast_1d(record_inds)
            color = 'cyan'
            spike_times = np.asarray(self.spikeMonExcT)[self.spikeMonExcI == neuron_index]
            thresh = float(self.p.get('vThreshExc', -52) / mV) if 'vThreshExc' in self.p else -52
        else:
            V = np.asarray(self.stateMonInhV)
            record_inds = self.p.get('record_voltage_inh_inds')
            record_inds = np.arange(V.shape[0]) if record_inds is None else np.atleast_1d(record_inds)
            color = 'red'
            spike_times = np.asarray(self.spikeMonInhT)[self.spikeMonInhI == neuron_index]
            thresh = float(self.p.get('vThreshInh', -43) / mV) if 'vThreshInh' in self.p else -43
        if mean:
            v = V.mean(axis=0)
            lw = 0.8
        else:
            row = np.where(record_inds == neuron_index)[0]
            if len(row) == 0:
                ax.set_title(f"Neuron {neuron_index} not in recorded set")
                return ax
            v = V[row[0]]
            lw = 0.6
        ax.plot(self.stateMonT, v, color=color, lw=lw)
        # Draw vertical spike markers for single-neuron trace
        if not mean and len(spike_times) > 0:
            for t in spike_times:
                ax.plot([t, t], [thresh, spike_peak_mV], color=color, lw=0.8, solid_capstyle='butt')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")

    def plot_voltage_by_groups(self, ax=None, use_sem=True):
        """Plot mean voltage ± error (SEM or SD) for each group (CS, US, NS, inh) on one axes."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        draw_cs_us_stimulus_spans(ax, self.p, zorder=0, legend_label=False)
        draw_cs_us_stimulus_pulse_lines(ax, self.p, zorder=1, legend_label=False)
        V_exc = np.asarray(self.stateMonExcV)
        V_inh = np.asarray(self.stateMonInhV)
        t = self.stateMonT
        # record_voltage_*_inds[i] = neuron index for row i in V (so we can map group indices to rows)
        rec_exc = self.p.get('record_voltage_exc_inds')
        rec_inh = self.p.get('record_voltage_inh_inds')
        if rec_exc is None:
            rec_exc = np.arange(V_exc.shape[0])
        if rec_inh is None:
            rec_inh = np.arange(V_inh.shape[0])
        rec_exc = np.atleast_1d(rec_exc)
        rec_inh = np.atleast_1d(rec_inh)

        had_any = [False]  # use list so inner function can set it

        def plot_group(neuron_inds, V, record_inds, color, label):
            # Map neuron indices to row indices: row k in V corresponds to neuron record_inds[k]
            neuron_inds = np.atleast_1d(neuron_inds)
            row_inds = np.where(np.isin(record_inds, neuron_inds))[0]
            if len(row_inds) == 0:
                return
            sub = V[row_inds]
            mean = sub.mean(axis=0)
            if mean.size == 0 or len(mean) != len(t):
                return
            n = sub.shape[0]
            err = sub.std(axis=0) / (np.sqrt(n) if use_sem else 1.0) if n > 1 else np.zeros_like(mean)
            ax.plot(t, mean, color=color, lw=0.8, label=label)
            ax.fill_between(t, mean - err, mean + err, color=color, alpha=0.3)
            had_any[0] = True

        if 'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p:
            cs_inds = np.atleast_1d(self.p['cs_neuron_inds'])
            us_inds = np.atleast_1d(self.p['us_neuron_inds'])
            nExc = self.p['nExc']
            ns_inds = np.array([i for i in range(nExc) if i not in cs_inds and i not in us_inds])
            plot_group(cs_inds, V_exc, rec_exc, 'C3', 'CS')
            plot_group(us_inds, V_exc, rec_exc, 'C0', 'US')
            plot_group(ns_inds, V_exc, rec_exc, '0.6', 'NS')
            plot_group(rec_inh, V_inh, rec_inh, 'red', 'inh')
        else:
            plot_group(rec_exc, V_exc, rec_exc, 'cyan', 'exc')
            plot_group(rec_inh, V_inh, rec_inh, 'red', 'inh')
        # Mark times when US would have started on CS-only trials (no US delivered)
        us_omit = self.p.get('us_omit_times_s')
        if us_omit is not None and len(us_omit) > 0:
            for t in np.atleast_1d(us_omit):
                ax.axvline(t, color='C0', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        if had_any[0]:
            ax.legend(loc='upper right', fontsize=7)
        return ax

    def plot_pca_3d_time_color(self, ax=None, bin_size=5*ms, use_exc_only=True,
                               line_alpha=0.85, line_lw=0.8, cmap='viridis'):
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        X = compute_population_matrix(self, bin_size, use_exc_only, subtract_mean=False)
        X_use = X - X.mean(axis=0)
        centers_use = centers

        if X_use.shape[0] < 2:
            if ax is None:
                fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            ax.set_title("First 3 PCs (insufficient time bins)")
            return ax
        _, _, Vt = np.linalg.svd(X_use, full_matrices=False)
        proj = X_use @ Vt[:3].T
        for j in range(3):
            col = proj[:, j]
            proj[:, j] = (col - col.mean()) / (col.std() + 1e-10)
        pc1, pc2, pc3 = proj[:, 0], proj[:, 1], proj[:, 2]
        t_plot = centers_use

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        norm = Normalize(vmin=t_plot.min(), vmax=t_plot.max())
        for i in range(len(pc1) - 1):
            c = plt.get_cmap(cmap)(norm((t_plot[i] + t_plot[i + 1]) / 2))
            ax.plot(pc1[i:i+2], pc2[i:i+2], pc3[i:i+2], color=c, alpha=line_alpha, lw=line_lw)
        sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        sm.set_array([])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("First 3 PCs (full simulation)")
        plt.colorbar(sm, ax=ax, shrink=0.6, label="Time (s)")
        return ax

    def plot_within_between_correlations(self, ax=None, bin_size=5*ms):
        cor = compute_within_between_correlations(self, bin_size)
        if ax is None:
            fig, ax = plt.subplots()
        if cor is None:
            ax.set_title("Within / between correlation (CS/US not defined)")
            return ax
        labels = ['within CS', 'within US', 'within NS', 'between CS–US', 'between CS–NS', 'between US–NS']
        values = [cor['within_CS'], cor['within_US'], cor['within_NS'],
                  cor['between_CS_US'], cor['between_CS_NS'], cor['between_US_NS']]
        colors = ['C3', 'C0', '0.6', 'purple', 'brown', 'green']
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel("Mean correlation")
        ax.set_title("Within vs between group correlation (exc)")
        return ax

    def plot_pca_variance(self, ax=None, bin_size=5*ms, use_exc_only=True,
                         max_components=50, bar=True):
        var_pct = compute_pca_variance_explained(self, bin_size, use_exc_only)
        n_show = min(max_components, len(var_pct))
        var_pct = var_pct[:n_show]
        if ax is None:
            fig, ax = plt.subplots()
        x = np.arange(1, len(var_pct) + 1)
        if bar:
            ax.bar(x, var_pct, color='steelblue', alpha=0.8, edgecolor='navy')
        else:
            ax.plot(x, var_pct, 'o-', color='steelblue', markersize=3)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance explained (%)")
        ax.set_title("PCA variance explained (full simulation)")
        return ax

    def plot_pca_centroid_trajectories(self, bin_size=5*ms, n_components=3,
                                       ax_traj=None, ax_dist=None, use_exc_only=True):
        """
        Build trial-separated binned data, fit PCA on all trials, project and compute CS/US centroids,
        then plot centroid trajectories (3D PC1–PC2–PC3) and centroid distance over time.
        Requires params: trial_starts_s, trial_duration_s, trial_conditions.
        Returns the figure if one was created, else None.
        """
        data, conditions = compute_trial_binned_data(self, bin_size, use_exc_only)
        if data is None or data.size == 0:
            return None
        # Need both conditions to have at least one trial
        uniq = np.unique(conditions)
        if len(uniq) < 2:
            return None
        projected_trials, centroid_CS, centroid_US, centroid_distance = pca_condition_trajectories(
            data, conditions, n_components, condition_CS='CS', condition_US='US'
        )
        bin_size_s = float(bin_size / second)
        trial_duration_s = self.p['trial_duration_s']
        n_timepoints = centroid_CS.shape[0]
        time_s = (np.arange(n_timepoints) + 0.5) * bin_size_s

        if ax_traj is None and ax_dist is None:
            fig = plt.figure(figsize=(12, 5))
            ax_traj = fig.add_subplot(1, 2, 1, projection='3d')
            ax_dist = fig.add_subplot(1, 2, 2)
            created_fig = fig
        else:
            created_fig = None
            if ax_traj is None:
                ax_traj = plt.gca()
            if ax_dist is None:
                ax_dist = plt.gca()

        # Left: 3D trajectory (PC1, PC2, PC3)
        for ax_cur, cent, label, color in [
            (ax_traj, centroid_CS, 'CS', 'C3'),
            (ax_traj, centroid_US, 'US', 'C0'),
        ]:
            if np.any(np.isfinite(cent)):
                ax_cur.plot(cent[:, 0], cent[:, 1], cent[:, 2], color=color, lw=2, alpha=0.9, label=label)
                step = max(1, n_timepoints // 8)
                for k in range(0, n_timepoints, step):
                    ax_cur.scatter(cent[k, 0], cent[k, 1], cent[k, 2], c=[color], s=20, edgecolors='k', linewidths=0.5)
        ax_traj.set_xlabel("PC1")
        ax_traj.set_ylabel("PC2")
        ax_traj.set_zlabel("PC3")
        ax_traj.set_title("Condition centroid trajectories")
        ax_traj.legend(loc='best')

        # Right: centroid distance over time within trial
        ax_dist.plot(time_s, centroid_distance, 'k-', lw=1.5)
        ax_dist.fill_between(time_s, 0, centroid_distance, alpha=0.2)
        ax_dist.set_xlabel("Time within trial (s)")
        ax_dist.set_ylabel("CS–US centroid distance")
        ax_dist.set_title("Centroid distance over time")
        ax_dist.set_xlim(0, trial_duration_s)

        if created_fig:
            created_fig.tight_layout()
        return created_fig

    def get_mean_firing_rates(self):
        """
        Population-mean firing rates (Hz) for E and I over params['duration'].

        Returns
        -------
        dict or None
            Keys: mean_rate_E_Hz, mean_rate_I_Hz, duration_s, n_spikes_E, n_spikes_I, nExc, nInh.
        """
        return compute_mean_firing_rates(self)

    def plot_ns_trial_trajectories(self, bin_size=5*ms, n_components=3, ax_2d=None, ax_3d=None,
                                   color_by_time=True, cmap='viridis', alpha=0.75, linewidth=1.2):
        """
        Plot the NS population trajectory in PCA space for the full simulation (one continuous path).
        Color indicates simulation time (early→late) when color_by_time is True.

        Parameters
        ----------
        bin_size : Quantity
            Time bin for binned firing rates.
        n_components : int
            Number of PCA components (2 or 3 for plotting).
        ax_2d, ax_3d : matplotlib axes, optional
            If provided, draw in these axes (ax_2d = PC1 vs PC2, ax_3d = 3D). If both None, create a figure with 2 panels.
        color_by_time : bool
            If True, color by simulation time; else use a single color.
        cmap : str
            Colormap name for time (e.g. 'viridis', 'plasma').
        alpha, linewidth : float
            Line transparency and width.

        Returns
        -------
        fig or None
            The figure if one was created, else None.
        """
        out = compute_ns_trial_trajectories(self, bin_size=bin_size, n_components=max(n_components, 2))
        projected, time_s, _conditions, pca, ns_inds = out
        if projected is None or pca is None:
            return None
        n_timepoints, nc = projected.shape
        if nc < 2:
            return None

        create_fig = (ax_2d is None and ax_3d is None)
        if create_fig:
            fig = plt.figure(figsize=(12, 5))
            ax_2d = fig.add_subplot(1, 2, 1)
            ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
            fig_out = fig
        else:
            fig_out = None
            if ax_2d is None:
                ax_2d = plt.gca()
            if ax_3d is None:
                ax_3d = None

        norm = Normalize(vmin=time_s.min(), vmax=time_s.max())
        cmap_obj = plt.get_cmap(cmap)

        for t in range(n_timepoints - 1):
            c = cmap_obj(norm((time_s[t] + time_s[t + 1]) / 2)) if color_by_time else '0.5'
            ax_2d.plot(projected[t:t + 2, 0], projected[t:t + 2, 1], color=c, alpha=alpha, lw=linewidth)
        if ax_3d is not None and nc >= 3:
            for t in range(n_timepoints - 1):
                c = cmap_obj(norm((time_s[t] + time_s[t + 1]) / 2)) if color_by_time else '0.5'
                ax_3d.plot(
                    projected[t:t + 2, 0],
                    projected[t:t + 2, 1],
                    projected[t:t + 2, 2],
                    color=c,
                    alpha=alpha,
                    lw=linewidth,
                )

        ax_2d.set_xlabel("PC1 (NS)")
        ax_2d.set_ylabel("PC2 (NS)")
        ax_2d.set_title("NS population trajectory (full simulation)")
        ax_2d.set_aspect('equal', adjustable='datalim')
        ax_2d.axhline(0, color='k', lw=0.3, alpha=0.5)
        ax_2d.axvline(0, color='k', lw=0.3, alpha=0.5)
        if color_by_time:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])
            plt.colorbar(sm, ax=ax_2d, shrink=0.7, label="Time (s)")

        if ax_3d is not None and nc >= 3:
            ax_3d.set_xlabel("PC1 (NS)")
            ax_3d.set_ylabel("PC2 (NS)")
            ax_3d.set_zlabel("PC3 (NS)")
            ax_3d.set_title("NS trajectory 3D (full simulation)")
            if color_by_time:
                plt.colorbar(sm, ax=ax_3d, shrink=0.6, label="Time (s)")

        if create_fig and fig_out is not None:
            fig_out.tight_layout()
        return fig_out

    def plot_readout_evaluations(self, bin_size=10*ms, max_lag_bins=40, axes=None):
        """
        Evaluation panel for reservoir/readout goals:
          1) Time-resolved NS decodability (CS vs US trials),
          2) Reservoir memory curve (lagged CS pulse reconstruction R^2),
          3) Trial-level US readout score with AUC / d'.
        Returns the created figure, or None if custom axes were provided.
        """
        created_fig = None
        if axes is None:
            created_fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax_dec, ax_mem, ax_us = axes

        t_dec, acc = compute_ns_time_resolved_decodability(self, bin_size=bin_size, n_splits=5)
        if t_dec is None or acc is None:
            ax_dec.text(0.5, 0.5, "NS decodability unavailable", ha='center', va='center', transform=ax_dec.transAxes)
            ax_dec.set_xticks([])
            ax_dec.set_yticks([])
        else:
            ax_dec.plot(t_dec, acc, color='C2', lw=1.6)
            ax_dec.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.7)
            ax_dec.set_ylim(0.0, 1.0)
            ax_dec.set_xlabel("Time within trial (s)")
            ax_dec.set_ylabel("CV accuracy")
            ax_dec.set_title("NS decodability (CS vs US)")

        lags_s, r2 = compute_reservoir_memory_curve(self, bin_size=bin_size, max_lag_bins=max_lag_bins)
        if lags_s is None or r2 is None:
            ax_mem.text(0.5, 0.5, "Memory curve unavailable", ha='center', va='center', transform=ax_mem.transAxes)
            ax_mem.set_xticks([])
            ax_mem.set_yticks([])
        else:
            ax_mem.plot(lags_s, r2, color='C1', marker='o', ms=3, lw=1.2)
            ax_mem.axhline(0.0, color='k', lw=0.8, ls='--', alpha=0.7)
            ax_mem.set_xlabel("Lag (s)")
            ax_mem.set_ylabel(r"$R^2$")
            ax_mem.set_title("Reservoir memory curve")

        m = compute_us_readout_metrics(self)
        if m is None:
            ax_us.text(0.5, 0.5, "US readout metrics unavailable", ha='center', va='center', transform=ax_us.transAxes)
            ax_us.set_xticks([])
            ax_us.set_yticks([])
        else:
            cond = np.asarray(m['trial_conditions'])
            rates = np.asarray(m['trial_rates_Hz'], dtype=float)
            x = np.arange(rates.size)
            color = np.where(cond == 'US', 'C0', np.where(cond == 'CS', 'C3', '0.6'))
            ax_us.scatter(x, rates, c=color, s=20, alpha=0.85)
            w0, w1 = m['us_window_s']
            ax_us.set_xlabel("Trial")
            ax_us.set_ylabel("US-pop firing rate (Hz)")
            ax_us.set_title(
                "US readout: AUC={:.3f}, d'={:.3f}, acc={:.3f}\nWindow [{:.3f}, {:.3f}] s".format(
                    m['auc'], m['dprime'], m['threshold_acc'], w0, w1
                )
            )
            us_mask = (cond == 'US')
            cs_mask = (cond == 'CS')
            if np.any(us_mask):
                ax_us.axhline(np.mean(rates[us_mask]), color='C0', lw=1.0, ls='--', alpha=0.7)
            if np.any(cs_mask):
                ax_us.axhline(np.mean(rates[cs_mask]), color='C3', lw=1.0, ls='--', alpha=0.7)

        if created_fig is not None:
            created_fig.tight_layout()
        return created_fig

    def plot_ee_w_in_w_out(self, figsize=(8, 6)):
        """
        W_in and W_out vs time for CS, US, NS excitatory populations (Litwin-Kumar & Doiron 2014 definitions).
        Requires params populated by Network.run with record_ee_w_stats.
        Returns a new figure or None if no samples were recorded.
        """
        if self.w_stats_t.size == 0 or self.w_in_CS_US_NS.size == 0:
            return None
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        t = self.w_stats_t
        labels = ('CS', 'US', 'NS')
        colors = ('C3', 'C0', 'C2')
        win = self.w_in_CS_US_NS * 1e9  # siemens → nS
        wout = self.w_out_CS_US_NS * 1e9
        n_pop = min(3, win.shape[1])
        for k in range(n_pop):
            ax1.plot(
                t, np.where(np.isfinite(win[:, k]), win[:, k], np.nan),
                color=colors[k], lw=1.2, label=labels[k],
            )
            ax2.plot(
                t, np.where(np.isfinite(wout[:, k]), wout[:, k], np.nan),
                color=colors[k], lw=1.2, label=labels[k],
            )
        ax1.set_ylabel(r'$W_{\mathrm{in}}$ (nS)')
        ax1.set_title(r'$W_{\mathrm{in}}$: mean EE weight within population')
        ax1.legend(ncol=3, fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        ax2.set_ylabel(r'$W_{\mathrm{out}}$ (nS)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title(r'$W_{\mathrm{out}}$: mean EE weight between populations')
        ax2.legend(ncol=3, fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_weight_change_blocks(self, ax=None):
        """
        Bar graph of mean ± SEM percentage weight change by block (CS→CS, CS→US, US→CS, US→US, etc.)
        Uses EE block only with groups CS, US, NS. Requires weight_matrix_pre/post and cs/us_neuron_inds.
        Returns the figure if one was created, else None.
        """
        if 'weight_matrix_pre' not in self.p or 'weight_matrix_post' not in self.p:
            return None
        if 'cs_neuron_inds' not in self.p or 'us_neuron_inds' not in self.p:
            return None
        W_pre = np.asarray(self.p['weight_matrix_pre'])
        W_post = np.asarray(self.p['weight_matrix_post'])
        nExc = self.p['nExc']
        # EE block only
        W_pre_EE = W_pre[:nExc, :nExc]
        W_post_EE = W_post[:nExc, :nExc]
        cs_inds = np.atleast_1d(self.p['cs_neuron_inds'])
        us_inds = np.atleast_1d(self.p['us_neuron_inds'])
        ns_inds = np.array([i for i in range(nExc) if i not in cs_inds and i not in us_inds])
        groups = np.zeros(nExc, dtype=int)
        groups[cs_inds] = 0
        groups[us_inds] = 1
        groups[ns_inds] = 2
        group_names = ['CS', 'US', 'NS']
        labels, means, sems = compute_block_weight_change(
            W_pre_EE, W_post_EE, groups, group_names=group_names
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            created_fig = fig
        else:
            created_fig = None
        x = np.arange(len(labels))
        colors = ['C3', 'C0', '0.6']
        bar_colors = [colors[i // 3] if i // 3 < 3 else 'gray' for i in range(len(labels))]
        ax.bar(x, means, yerr=sems, capsize=4, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel("Weight change (%)")
        ax.set_title("EE weight change by block (mean ± SEM)")
        return created_fig


def plot_all_figures(results, show=True):
    """
    Create the standard poster figures from a SimpleResults instance.
    Returns (fig1, fig2, fig3, fig4, fig5, fig6, fig7). If show is True, calls plt.show() at the end.
    fig5: NS population PCA trajectory over the full simulation time.
    fig6: W_in / W_out time series for CS, US, NS (None if not recorded).
    fig7: Reservoir/readout evaluation panel (decodability, memory, US readout score).
    """
    # Figure 1: raster, firing rate, voltage
    fig1 = plt.figure(figsize=(8, 10))
    ax_raster = fig1.add_subplot(3, 1, 1)
    ax_rate = fig1.add_subplot(3, 1, 2)
    ax_voltage = fig1.add_subplot(3, 1, 3)
    results.plot_spike_raster(ax_raster)
    results.plot_firing_rate(ax_rate)
    results.plot_voltage_by_groups(ax_voltage)
    fig1.tight_layout()

    # Figure 2: PCA 3D, within/between correlation, PCA variance
    pca_bin_size = 10 * ms
    fig2 = plt.figure(figsize=(10, 12))
    gs = fig2.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax_pca = fig2.add_subplot(gs[0], projection='3d')
    ax_corr = fig2.add_subplot(gs[1])
    ax_var = fig2.add_subplot(gs[2])
    results.plot_pca_3d_time_color(ax=ax_pca, bin_size=pca_bin_size)
    results.plot_within_between_correlations(ax=ax_corr)
    results.plot_pca_variance(ax=ax_var, bin_size=pca_bin_size)
    fig2.tight_layout()

    # Figure 3: weight change by block
    fig3 = results.plot_weight_change_blocks()
    if fig3 is not None:
        fig3.tight_layout()

    # Figure 4: PCA centroid trajectories
    fig4 = results.plot_pca_centroid_trajectories(bin_size=pca_bin_size, n_components=3)
    if fig4 is not None:
        fig4.tight_layout()

    # Figure 5: NS population trial trajectories (compare trial-to-trial similarity)
    fig5 = results.plot_ns_trial_trajectories(bin_size=5*ms, n_components=3)
    if fig5 is not None:
        fig5.tight_layout()

    # Figure 6: L&D-style W_in / W_out for CS, US, NS
    fig6 = results.plot_ee_w_in_w_out()

    fig7 = results.plot_readout_evaluations(bin_size=10*ms, max_lag_bins=40)

    if show:
        plt.show()
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

