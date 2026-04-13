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


def draw_cs_us_stimulus_spans(ax, p, zorder=0, legend_label=True):
    """
    Shade CS/US intervals from params cs_stim_intervals_s, us_stim_intervals_s (each row [t0, t1] s).
    Sustained mode: short windows per nominal pulse; pulse_train mode: full epoch per trial.
    Trial-level envelopes (if needed): cs_stim_epoch_intervals_s, us_stim_epoch_intervals_s.
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
    Vertical line at each external CS/US input spike time (params cs_stim_pulse_times_s,
    us_stim_pulse_times_s from Network._build_cs_us_input).
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


def compute_pca_variance_explained(results, bin_size=5*ms, use_exc_only=True,
                                   use_upstate_only=False, upstate_bin_size=10*ms,
                                   p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0):
    """Variance explained by each PC (%). Returns 1d array."""
    X = compute_population_matrix(results, bin_size, use_exc_only, subtract_mean=False)
    if use_upstate_only:
        _, upstate_mask_coarse = detect_upstates(results, bin_size=upstate_bin_size, use_exc_only=use_exc_only,
                                                  p_stay=p_stay, rate_up_ratio=rate_up_ratio, rate_down_ratio=rate_down_ratio)
        upstate_bin_size_s = float(upstate_bin_size / second)
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, results.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        upstate_mask = np.zeros(len(centers), dtype=bool)
        for i in range(len(centers)):
            k = min(int(centers[i] / upstate_bin_size_s), len(upstate_mask_coarse) - 1)
            if k >= 0:
                upstate_mask[i] = upstate_mask_coarse[k]
        if np.any(upstate_mask):
            X = X[upstate_mask] - X[upstate_mask].mean(axis=0)
        # else X stays as-is; svd of empty will be handled below
    else:
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
    Project each trial's NS (non-stimulated) population activity into a shared PCA space
    for comparing trial-to-trial trajectory similarity.

    Requires params: trial_starts_s, trial_duration_s, trial_conditions, cs_neuron_inds, us_neuron_inds.

    Returns
    -------
    projected_trials : ndarray, shape (n_trials, n_timepoints, n_components)
        Each trial's NS trajectory in PCA space.
    time_s : ndarray, shape (n_timepoints,)
        Time within trial (s) for each bin.
    conditions : ndarray, shape (n_trials,)
        Condition label per trial ('CS' or 'US').
    pca : sklearn PCA
        Fitted PCA on (n_trials * n_timepoints, n_ns_neurons) NS data.
    ns_inds : ndarray
        Indices of NS neurons.
    or (None, None, None, None, None) if trial/CS/US info missing or no NS neurons.
    """
    p = results.p
    if 'trial_starts_s' not in p or 'trial_duration_s' not in p or 'trial_conditions' not in p:
        return None, None, None, None, None
    if 'cs_neuron_inds' not in p or 'us_neuron_inds' not in p:
        return None, None, None, None, None
    cs_set = set(np.atleast_1d(p['cs_neuron_inds']))
    us_set = set(np.atleast_1d(p['us_neuron_inds']))
    n_exc = p['nExc']
    ns_inds = np.array([i for i in range(n_exc) if i not in cs_set and i not in us_set])
    if len(ns_inds) == 0:
        return None, None, None, None, None

    data, conditions = compute_trial_binned_data(results, bin_size=bin_size, use_exc_only=True)
    if data is None or data.size == 0:
        return None, None, None, None, None
    # data: (n_trials, n_timepoints, n_neurons); restrict to NS
    ns_data = data[:, :, ns_inds]  # (n_trials, n_timepoints, n_ns)
    n_trials, n_timepoints, n_ns = ns_data.shape

    # Fit PCA on all NS activity (all trials, all timepoints)
    X_all = ns_data.reshape(-1, n_ns)
    n_components = min(n_components, n_ns, X_all.shape[0] - 1)
    if n_components < 1:
        return None, None, None, None, None
    pca = PCA(n_components=n_components)
    pca.fit(X_all)

    # Project each trial
    projected_trials = np.zeros((n_trials, n_timepoints, n_components))
    for tr in range(n_trials):
        projected_trials[tr] = pca.transform(ns_data[tr])

    bin_size_s = float(bin_size / second)
    time_s = (np.arange(n_timepoints) + 0.5) * bin_size_s
    return projected_trials, time_s, conditions, pca, ns_inds


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
            if nCS > 0:
                mask_cs = np.isin(self.spikeMonExcI, self.p['cs_neuron_inds'])
                t_cs = self.spikeMonExcT[mask_cs]
                FR_CS, _ = np.histogram(t_cs, bins)
                FR_CS = FR_CS / bin_size_s / nCS
                ax.plot(centers, FR_CS, color='C3', alpha=0.8, label='CS')
            if nUS > 0:
                mask_us = np.isin(self.spikeMonExcI, self.p['us_neuron_inds'])
                t_us = self.spikeMonExcT[mask_us]
                FR_US, _ = np.histogram(t_us, bins)
                FR_US = FR_US / bin_size_s / nUS
                ax.plot(centers, FR_US, color='C0', alpha=0.8, label='US')
            if nNS > 0:
                ns_set = set(range(nExc)) - cs_set - us_set
                mask_ns = np.isin(self.spikeMonExcI, np.array(list(ns_set)))
                t_ns = self.spikeMonExcT[mask_ns]
                FR_ns, _ = np.histogram(t_ns, bins)
                FR_ns = FR_ns / bin_size_s / nNS
                ax.plot(centers, FR_ns, color='0.6', alpha=0.6, label='NS')
            FRInh, _ = np.histogram(self.spikeMonInhT, bins)
            FRInh = FRInh / bin_size_s / nInh
            ax.plot(centers, FRInh, color='red', alpha=0.6, label='inh')
        else:
            FRExc, _ = np.histogram(self.spikeMonExcT, bins)
            FRInh, _ = np.histogram(self.spikeMonInhT, bins)
            FRExc = FRExc / bin_size_s / nExc
            FRInh = FRInh / bin_size_s / nInh
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
                               use_upstate_only=True, upstate_bin_size=10*ms,
                               p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0,
                               line_alpha=0.85, line_lw=0.8, cmap='viridis'):
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        X = compute_population_matrix(self, bin_size, use_exc_only, subtract_mean=False)

        if use_upstate_only:
            _, upstate_mask_coarse = detect_upstates(self, bin_size=upstate_bin_size, use_exc_only=use_exc_only,
                                                     p_stay=p_stay, rate_up_ratio=rate_up_ratio, rate_down_ratio=rate_down_ratio)
            upstate_bin_size_s = float(upstate_bin_size / second)
            upstate_mask = np.zeros(len(centers), dtype=bool)
            for i in range(len(centers)):
                k = min(int(centers[i] / upstate_bin_size_s), len(upstate_mask_coarse) - 1)
                if k >= 0:
                    upstate_mask[i] = upstate_mask_coarse[k]
            if not np.any(upstate_mask):
                X_use = np.empty((0, X.shape[1]))
                centers_use = np.array([])
            else:
                X_use = X[upstate_mask] - X[upstate_mask].mean(axis=0)
                centers_use = centers[upstate_mask]
            if X_use.shape[0] < 2:
                if ax is None:
                    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
                ax.set_title("First 3 PCs (upstate only; no upstate bins)")
                return ax
        else:
            X_use = X - X.mean(axis=0)
            centers_use = centers

        U, S, Vt = np.linalg.svd(X_use, full_matrices=False)
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
        ax.set_title("First 3 PCs (upstate only)" if use_upstate_only else "First 3 PCs (full simulation)")
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
                         use_upstate_only=False, upstate_bin_size=10*ms,
                         p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0,
                         max_components=50, bar=True):
        var_pct = compute_pca_variance_explained(
            self, bin_size, use_exc_only, use_upstate_only, upstate_bin_size,
            p_stay, rate_up_ratio, rate_down_ratio
        )
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
        title = "PCA variance explained (upstate only)" if use_upstate_only else "PCA variance explained (full)"
        ax.set_title(title)
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
        Plot each trial's NS population trajectory in PCA space to compare trial-to-trial similarity.
        Trajectories are overlaid; color indicates time within trial (early→late) so direction is visible.
        Similar trials will overlap; different trials will diverge.

        Parameters
        ----------
        bin_size : Quantity
            Time bin for binned firing rates.
        n_components : int
            Number of PCA components (2 or 3 for plotting).
        ax_2d, ax_3d : matplotlib axes, optional
            If provided, draw in these axes (ax_2d = PC1 vs PC2, ax_3d = 3D). If both None, create a figure with 2 panels.
        color_by_time : bool
            If True, color trajectory by time within trial; else use a single color per trial.
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
        projected_trials, time_s, conditions, pca, ns_inds = out
        if projected_trials is None or pca is None:
            return None
        n_trials, n_timepoints, nc = projected_trials.shape
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

        for tr in range(n_trials):
            proj = projected_trials[tr]  # (n_timepoints, n_components)
            for t in range(n_timepoints - 1):
                c = cmap_obj(norm((time_s[t] + time_s[t + 1]) / 2)) if color_by_time else '0.5'
                ax_2d.plot(proj[t:t + 2, 0], proj[t:t + 2, 1], color=c, alpha=alpha, lw=linewidth)
            if ax_3d is not None and nc >= 3:
                for t in range(n_timepoints - 1):
                    c = cmap_obj(norm((time_s[t] + time_s[t + 1]) / 2)) if color_by_time else '0.5'
                    ax_3d.plot(proj[t:t + 2, 0], proj[t:t + 2, 1], proj[t:t + 2, 2], color=c, alpha=alpha, lw=linewidth)

        ax_2d.set_xlabel("PC1 (NS)")
        ax_2d.set_ylabel("PC2 (NS)")
        ax_2d.set_title("NS population trajectory (each line = one trial)")
        ax_2d.set_aspect('equal', adjustable='datalim')
        ax_2d.axhline(0, color='k', lw=0.3, alpha=0.5)
        ax_2d.axvline(0, color='k', lw=0.3, alpha=0.5)
        if color_by_time:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])
            plt.colorbar(sm, ax=ax_2d, shrink=0.7, label="Time in trial (s)")

        if ax_3d is not None and nc >= 3:
            ax_3d.set_xlabel("PC1 (NS)")
            ax_3d.set_ylabel("PC2 (NS)")
            ax_3d.set_zlabel("PC3 (NS)")
            ax_3d.set_title("NS trajectory 3D (each line = one trial)")
            if color_by_time:
                plt.colorbar(sm, ax=ax_3d, shrink=0.6, label="Time in trial (s)")

        if create_fig and fig_out is not None:
            fig_out.tight_layout()
        return fig_out

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
    Returns (fig1, fig2, fig3, fig4, fig5, fig6). If show is True, calls plt.show() at the end.
    fig5: NS population trial trajectories (per-trial PCA trajectory overlay).
    fig6: W_in / W_out time series for CS, US, NS (None if not recorded).
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
    fig2 = plt.figure(figsize=(10, 12))
    gs = fig2.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax_pca = fig2.add_subplot(gs[0], projection='3d')
    ax_corr = fig2.add_subplot(gs[1])
    ax_var = fig2.add_subplot(gs[2])
    results.plot_pca_3d_time_color(ax=ax_pca, use_upstate_only=True)
    results.plot_within_between_correlations(ax=ax_corr)
    results.plot_pca_variance(ax=ax_var, use_upstate_only=True)
    fig2.tight_layout()

    # Figure 3: weight change by block
    fig3 = results.plot_weight_change_blocks()
    if fig3 is not None:
        fig3.tight_layout()

    # Figure 4: PCA centroid trajectories
    fig4 = results.plot_pca_centroid_trajectories(bin_size=10*ms, n_components=3)
    if fig4 is not None:
        fig4.tight_layout()

    # Figure 5: NS population trial trajectories (compare trial-to-trial similarity)
    fig5 = results.plot_ns_trial_trajectories(bin_size=5*ms, n_components=3)
    if fig5 is not None:
        fig5.tight_layout()

    # Figure 6: L&D-style W_in / W_out for CS, US, NS
    fig6 = results.plot_ee_w_in_w_out()

    if show:
        plt.show()
    return fig1, fig2, fig3, fig4, fig5, fig6

