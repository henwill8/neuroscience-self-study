"""
Poster-focused plotting: full-run spike raster, low-pass firing rates (first/last trial), EE ΔW blocks,
W_in/W_out vs time, trial-averaged PCA. Computation helpers live at module level; SimpleResults holds wrappers.
"""
import numpy as np
from sklearn.decomposition import PCA
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# Poster / print column: square figures, large type, high-contrast colormaps
POSTER_COL_INCH = 7.5
# Readable when printed large or viewed at a distance (~18–24" poster column)
POSTER_LABEL_FS = 20
POSTER_TICK_FS = 16
POSTER_TITLE_FS = 22
POSTER_LINEWIDTH = 2.2
# 'turbo' keeps a clear hue progression without viridis's low-contrast yellow-green at high values
POSTER_PCA_CMAP = 'turbo'
POSTER_PCA_LINEWIDTH = 3.2
# Scatter `s` (points^2); large enough that spikes stay visible when a trial spans a narrow x-range
POSTER_RASTER_SCATTER = 15.0


def configure_poster_matplotlib():
    """
    Sans-serif stack and mathtext defaults for legible poster export (PDF/PNG).
    Call is idempotent; run once when this module is imported.
    """
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': [
                'Arial',
                'Helvetica Neue',
                'Helvetica',
                'Nimbus Sans',
                'DejaVu Sans',
                'Bitstream Vera Sans',
                'sans-serif',
            ],
            'font.size': POSTER_TICK_FS,
            'axes.labelsize': POSTER_LABEL_FS,
            'axes.titlesize': POSTER_TITLE_FS,
            'axes.labelweight': 'normal',
            'axes.titleweight': 'bold',
            'xtick.labelsize': POSTER_TICK_FS,
            'ytick.labelsize': POSTER_TICK_FS,
            'legend.fontsize': POSTER_TICK_FS,
            'figure.titlesize': POSTER_TITLE_FS,
            'mathtext.fontset': 'dejavusans',
        }
    )


configure_poster_matplotlib()


# ---------------------------------------------------------------------------
# Low-pass + trial-averaged PCA (population exc by default)
# ---------------------------------------------------------------------------


def _pca_lowpass_tau_s(results, tau_s=None):
    """Time constant (seconds) for exponential low-pass on PCA spike-derived rates."""
    if tau_s is not None:
        return float(tau_s)
    p = results.p
    tau = p.get('pca_lowpass_tau', 50 * ms)
    return float(tau / second)


def _lowpass_instantaneous_rates(inst_rates, dt_s, tau_s):
    """
    First-order low-pass along time (causal): y[t] = a*y[t-1] + (1-a)*r[t], a = exp(-dt/tau).
    inst_rates: (n_timepoints, n_neurons) instantaneous rates in Hz.
    """
    inst_rates = np.asarray(inst_rates, dtype=float)
    if inst_rates.size == 0:
        return inst_rates
    dt_s = float(dt_s)
    tau_s = max(float(tau_s), 1e-9)
    a = np.exp(-dt_s / tau_s)
    y = np.zeros_like(inst_rates)
    y[0] = inst_rates[0]
    for k in range(1, inst_rates.shape[0]):
        y[k] = a * y[k - 1] + (1.0 - a) * inst_rates[k]
    return y


def compute_full_lowpass_population_matrix(
    results, bin_size=5 * ms, use_exc_only=True, subtract_mean=False, tau_s=None
):
    """
    Full-run instantaneous rate per bin (Hz), then causal low-pass per neuron.
    Shape (n_bins, n_neurons).
    """
    bin_size_s = float(bin_size / second)
    tau_s = _pca_lowpass_tau_s(results, tau_s)
    bins = np.arange(0, results.duration, bin_size_s)
    n_bins = len(bins) - 1
    n_neurons = results.p['nExc'] if use_exc_only else results.p['nUnits']
    inst = np.zeros((n_bins, n_neurons))
    for i in range(results.p['nExc']):
        spikes = results.spikeMonExcT[results.spikeMonExcI == i]
        counts, _ = np.histogram(spikes, bins)
        inst[:, i] = counts.astype(float) / bin_size_s
    if not use_exc_only:
        for i in range(results.p['nInh']):
            spikes = results.spikeMonInhT[results.spikeMonInhI == i]
            counts, _ = np.histogram(spikes, bins)
            inst[:, results.p['nExc'] + i] = counts.astype(float) / bin_size_s
    X = _lowpass_instantaneous_rates(inst, bin_size_s, tau_s)
    if subtract_mean:
        X = X - X.mean(axis=0)
    return X


def compute_trial_lowpass_data(results, bin_size=5 * ms, use_exc_only=True, tau_s=None):
    """
    Trial-aligned instantaneous rates (Hz), then causal low-pass in time within each trial.
    Returns (data, conditions) with data shape (n_trials, n_timepoints, n_neurons), or (None, None).
    """
    p = results.p
    if 'trial_starts_s' not in p or 'trial_duration_s' not in p or 'trial_conditions' not in p:
        return None, None
    trial_starts_s = np.asarray(p['trial_starts_s'])
    trial_duration_s = float(p['trial_duration_s'])
    conditions = np.asarray(p['trial_conditions'])
    bin_size_s = float(bin_size / second)
    tau_s = _pca_lowpass_tau_s(results, tau_s)
    n_trials = len(trial_starts_s)
    n_timepoints = int(round(trial_duration_s / bin_size_s))
    if n_timepoints < 1:
        return None, None
    n_neurons = p['nExc'] if use_exc_only else p['nUnits']
    inst = np.zeros((n_trials, n_timepoints, n_neurons))
    for tr in range(n_trials):
        t0 = float(trial_starts_s[tr])
        t1 = t0 + trial_duration_s
        bins = np.linspace(t0, t1, n_timepoints + 1)
        for i in range(p['nExc']):
            spikes = results.spikeMonExcT[results.spikeMonExcI == i]
            counts, _ = np.histogram(spikes, bins)
            inst[tr, :, i] = counts[:n_timepoints].astype(float) / bin_size_s
        if not use_exc_only:
            for i in range(p['nInh']):
                spikes = results.spikeMonInhT[results.spikeMonInhI == i]
                counts, _ = np.histogram(spikes, bins)
                inst[tr, :, p['nExc'] + i] = counts[:n_timepoints].astype(float) / bin_size_s
        inst[tr] = _lowpass_instantaneous_rates(inst[tr], bin_size_s, tau_s)
    return inst, conditions


def compute_single_trial_lowpass_matrix(results, trial_index, bin_size=5 * ms, tau_s=None):
    """
    Exponentially low-passed firing rate (Hz) for one trial, all E and I units.
    Returns (time_in_trial_s, rates) with rates shape (n_timepoints, nExc + nInh), or (None, None).
    """
    p = results.p
    if 'trial_starts_s' not in p or 'trial_duration_s' not in p:
        return None, None
    starts = np.asarray(p['trial_starts_s'], dtype=float).ravel()
    n_trials = int(starts.size)
    if n_trials == 0:
        return None, None
    tr = int(trial_index)
    if tr < 0:
        tr += n_trials
    if tr < 0 or tr >= n_trials:
        return None, None
    trial_duration_s = float(p['trial_duration_s'])
    bin_size_s = float(bin_size / second)
    tau_s = _pca_lowpass_tau_s(results, tau_s)
    n_timepoints = int(round(trial_duration_s / bin_size_s))
    if n_timepoints < 1:
        return None, None
    t0 = float(starts[tr])
    t1 = t0 + trial_duration_s
    bins = np.linspace(t0, t1, n_timepoints + 1)
    n_exc = int(p['nExc'])
    n_inh = int(p['nInh'])
    inst = np.zeros((n_timepoints, n_exc + n_inh))
    for i in range(n_exc):
        spikes = results.spikeMonExcT[results.spikeMonExcI == i]
        counts, _ = np.histogram(spikes, bins)
        inst[:, i] = counts[:n_timepoints].astype(float) / bin_size_s
    for i in range(n_inh):
        spikes = results.spikeMonInhT[results.spikeMonInhI == i]
        counts, _ = np.histogram(spikes, bins)
        inst[:, n_exc + i] = counts[:n_timepoints].astype(float) / bin_size_s
    rates = _lowpass_instantaneous_rates(inst, bin_size_s, tau_s)
    time_in_trial_s = (np.arange(n_timepoints) + 0.5) * (trial_duration_s / n_timepoints)
    return time_in_trial_s, rates


def compute_trial_avg_lowpass_matrix(results, bin_size=5 * ms, use_exc_only=True, tau_s=None):
    """
    Mean of trial-aligned low-pass population activity across trials: (n_timepoints, n_neurons).
    None if trial structure unavailable.
    """
    data, _conds = compute_trial_lowpass_data(results, bin_size, use_exc_only, tau_s)
    if data is None:
        return None
    return np.mean(np.asarray(data, dtype=float), axis=0)


def compute_pca_mean_trajectory_projected(
    results, bin_size=5 * ms, use_exc_only=True, n_components=3, tau_s=None
):
    """
    PCA on trial-averaged low-pass population activity; trajectory in PC space.
    Time axis is within-trial when trials are defined, else full-run bin centers.

    Returns
    -------
    time_s : (n_time,)
    proj : (n_time, n_components)
    pca : sklearn PCA or None if insufficient data
    """
    bin_size_s = float(bin_size / second)
    X_mean = compute_trial_avg_lowpass_matrix(results, bin_size, use_exc_only, tau_s=tau_s)
    if X_mean is not None and X_mean.shape[0] >= 2:
        n_tp = X_mean.shape[0]
        time_s = (np.arange(n_tp) + 0.5) * bin_size_s
    else:
        X_mean = compute_full_lowpass_population_matrix(
            results, bin_size, use_exc_only, subtract_mean=False, tau_s=tau_s
        )
        n_tp = X_mean.shape[0]
        bins = np.arange(0, results.duration, bin_size_s)
        time_s = bins[:-1] + bin_size_s / 2.0
    n_comp = min(int(n_components), X_mean.shape[1], X_mean.shape[0] - 1)
    if n_comp < 1:
        z = np.zeros((n_tp, max(int(n_components), 1)))
        return time_s, z, None
    pca = PCA(n_components=n_comp)
    pca.fit(X_mean)
    proj = pca.transform(X_mean)
    if n_comp < n_components:
        pad = np.zeros((proj.shape[0], n_components - n_comp))
        proj = np.hstack([proj, pad])
    return time_s, proj, pca


def _project_points_for_view(points_xyz, elev_deg, azim_deg, return_rotated=False):
    """
    Orthographic 3D->2D projection after applying view rotations.
    Returns projected (x2d, y2d). If return_rotated=True, also returns rotated xyz.
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        empty2 = np.zeros((0, 2), dtype=float)
        if return_rotated:
            return empty2, np.zeros((0, 3), dtype=float)
        return empty2
    # Matplotlib-like camera convention (approx): azimuth about z, elevation about x.
    az = np.deg2rad(float(azim_deg))
    el = np.deg2rad(float(elev_deg))
    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(el), np.sin(el)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot = rx @ rz
    pr = pts @ rot.T
    if return_rotated:
        return pr[:, :2], pr
    return pr[:, :2]


def suggest_pca_view_angles(points_xyz, elev_candidates=None, azim_candidates=None):
    """
    Choose a view (elev, azim) that minimizes projected trajectory overlap.
    Primary objective: fewest self-overlaps/intersections in 2D projection.
    Secondary objective: stronger perceived depth separation.
    Tertiary objective: larger nearest-neighbor spacing and spread for readability.
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return 22.0, 125.0
    if elev_candidates is None:
        elev_candidates = np.linspace(12.0, 40.0, 8)
    if azim_candidates is None:
        azim_candidates = np.linspace(0.0, 350.0, 36)

    def _segment_intersections_count(poly_xy):
        """
        Count strict intersections between non-adjacent segments of a polyline.
        """
        p = np.asarray(poly_xy, dtype=float)
        n = p.shape[0]
        if n < 4:
            return 0

        def _orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def _intersects(a, b, c, d):
            o1 = _orient(a, b, c)
            o2 = _orient(a, b, d)
            o3 = _orient(c, d, a)
            o4 = _orient(c, d, b)
            # Strict crossing only (ignore colinear-touch cases for stability).
            return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

        count = 0
        for i in range(n - 1):
            a, b = p[i], p[i + 1]
            for j in range(i + 2, n - 1):
                # Adjacent segments share a vertex; skip.
                if j == i:
                    continue
                c, d = p[j], p[j + 1]
                if _intersects(a, b, c, d):
                    count += 1
        return count

    best_score = -np.inf
    best_view = (22.0, 125.0)
    for elev in elev_candidates:
        for azim in azim_candidates:
            xy, rot_xyz = _project_points_for_view(pts, elev, azim, return_rotated=True)
            x = xy[:, 0]
            y = xy[:, 1]
            # Overlap metric:
            # 1) Minimize self-intersections in projected trajectory.
            n_intersections = _segment_intersections_count(xy)
            # Depth metric from camera-space z spread (larger => clearer depth).
            depth_spread = float(np.std(rot_xyz[:, 2])) if rot_xyz.shape[0] else 0.0
            # 2) Prefer larger local spacing to reduce visual overlap.
            diffs = np.diff(xy, axis=0)
            step_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
            step_med = float(np.median(step_lengths)) if step_lengths.size else 0.0
            # 3) Spread tie-breaker so the path fills the panel.
            spread = float(np.std(x) * np.std(y))
            # Large penalty on intersections; then depth; then spacing; then spread.
            score = (
                (-1000.0 * float(n_intersections))
                + (100.0 * depth_spread)
                + (10.0 * step_med)
                + spread
            )
            if score > best_score:
                best_score = score
                best_view = (float(elev), float(azim))
    return best_view


def compute_mean_firing_rates(results):
    """
    Population-mean firing rates (Hz) for excitatory and inhibitory units over the full run.
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


def compute_block_weight_change(W_pre, W_post, groups, group_names=None):
    """
    Block-averaged weight change: for each (pre, post) connection block, mean and SEM
    of percentage change (W_post - W_pre) / W_pre * 100 over synapses with W_pre > 0.
    Weight matrix convention: W[post, pre] = weight from pre to post.
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
            mask_post = groups == g_post
            mask_pre = groups == g_pre
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
# SimpleResults: data container + poster plots
# ---------------------------------------------------------------------------


class SimpleResults:
    """Holds spike/voltage data and params."""

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
        n_times = self.stateMonExcV.shape[1]
        self.stateMonT = np.arange(n_times, dtype=float) * self.stateDT
        self.w_stats_t = np.asarray(params.get('w_stats_t', []), dtype=float)
        self.w_in_CS_US_NS = np.asarray(params.get('w_in_CS_US_NS', np.zeros((0, 3))), dtype=float)
        self.w_out_CS_US_NS = np.asarray(params.get('w_out_CS_US_NS', np.zeros((0, 3))), dtype=float)

    def plot_spike_raster(
        self,
        ax,
        group_separators=True,
        t_window_s=None,
        time_relative_to_window=False,
        relative_time_origin_s=None,
        title=None,
        scatter_size=None,
    ):
        """
        Excitatory + inhibitory spike raster (CS | US | NS ordering when assembly indices exist).

        Parameters
        ----------
        t_window_s : (t0, t1) or None
            If set, only spikes with t0 <= t < t1 (simulation seconds). If None, full run.
        time_relative_to_window : bool
            If True and t_window_s is set, x-axis is time since t0 (0 … t1 - t0).
        title : str or None
            Optional axes title (e.g. trial label for poster panels).
        scatter_size : float or None
            Matplotlib scatter marker area (``s``). None → :data:`POSTER_RASTER_SCATTER`.
        """
        sdot = POSTER_RASTER_SCATTER if scatter_size is None else float(scatter_size)
        nExc = self.p['nExc']
        t_exc = np.asarray(self.spikeMonExcT)
        i_exc = np.asarray(self.spikeMonExcI)
        t_inh = np.asarray(self.spikeMonInhT)
        i_inh = np.asarray(self.spikeMonInhI)
        t0_win = t1_win = None
        if t_window_s is not None:
            t0_win, t1_win = float(t_window_s[0]), float(t_window_s[1])
            m_e = (t_exc >= t0_win) & (t_exc < t1_win)
            m_i = (t_inh >= t0_win) & (t_inh < t1_win)
            t_exc, i_exc = t_exc[m_e], i_exc[m_e]
            t_inh, i_inh = t_inh[m_i], i_inh[m_i]
            if time_relative_to_window:
                origin = t0_win if relative_time_origin_s is None else float(relative_time_origin_s)
                t_exc = t_exc - origin
                t_inh = t_inh - origin
        if 'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p:
            cs_set = set(self.p['cs_neuron_inds'])
            us_set = set(self.p['us_neuron_inds'])
            cs_sorted = np.sort(self.p['cs_neuron_inds'])
            us_sorted = np.sort(self.p['us_neuron_inds'])
            ns_sorted = np.sort([i for i in range(nExc) if i not in cs_set and i not in us_set])
            order = np.concatenate([cs_sorted, us_sorted, ns_sorted])
            neuron_to_display = {n: i for i, n in enumerate(order)}
            nCS, nUS = len(cs_sorted), len(us_sorted)
            y_exc = np.array([neuron_to_display[i] for i in i_exc])
            ax.scatter(t_exc, y_exc, c='#1f77b4', s=sdot, marker='.', linewidths=0, rasterized=True)
            if group_separators:
                ax.axhline(nCS - 0.5, color='0.70', lw=1.8)
                ax.axhline(nCS + nUS - 0.5, color='0.70', lw=1.8)
            ax.scatter(t_inh, nExc + i_inh, s=sdot, c='#d62728', marker='.', linewidths=0, rasterized=True)
            ax.axhline(nExc - 0.5, color='0.45', lw=1.4)
            ax.set_ylim(-0.5, self.p['nUnits'] - 0.5)
            nNS = len(ns_sorted)
            nInh = int(self.p.get('nInh', self.p['nUnits'] - nExc))
            yticks = []
            ylabels = []
            if nCS > 0:
                yticks.append((nCS - 1) / 2.0)
                ylabels.append('CS')
            if nUS > 0:
                yticks.append(nCS + (nUS - 1) / 2.0)
                ylabels.append('US')
            if nNS > 0:
                yticks.append(nCS + nUS + (nNS - 1) / 2.0)
                ylabels.append('NS')
            if nInh > 0:
                yticks.append(nExc + (nInh - 1) / 2.0)
                ylabels.append('I')
            if yticks:
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
            ax.set_ylabel('Group')
        else:
            ax.scatter(t_exc, i_exc, s=sdot, c='#17becf', marker='.', rasterized=True)
            ax.scatter(t_inh, nExc + i_inh, s=sdot, c='#d62728', marker='.', linewidths=0, rasterized=True)
            ax.set_ylim(-0.5, self.p['nUnits'] - 0.5)
            ax.set_ylabel('Neuron index')
        if t_window_s is None:
            ax.set_xlim(0, self.duration)
            ax.set_xlabel('Time (s)', fontsize=POSTER_LABEL_FS)
        elif time_relative_to_window:
            origin = t0_win if relative_time_origin_s is None else float(relative_time_origin_s)
            ax.set_xlim(t0_win - origin, t1_win - origin)
            ax.set_xlabel('Time in trial (s)', fontsize=POSTER_LABEL_FS)
        else:
            ax.set_xlim(t0_win, t1_win)
            ax.set_xlabel('Time (s)', fontsize=POSTER_LABEL_FS)
        ax.set_ylabel(ax.get_ylabel(), fontsize=POSTER_LABEL_FS)
        ax.tick_params(axis='both', which='major', labelsize=POSTER_TICK_FS)
        if title:
            ax.set_title(title, fontsize=POSTER_TITLE_FS, fontweight='bold', pad=10, loc='center')

    def plot_spike_raster_trial(
        self,
        trial_index,
        ax=None,
        figsize=None,
        group_separators=True,
        title=None,
        scatter_size=None,
        pre_silence_s=0.1,
        post_silence_s=0.1,
    ):
        """
        Spike raster for one trial; x-axis is time within that trial (0 … trial_duration_s).

        trial_index may be negative (e.g. -1 for last trial). Returns a new figure if ax is None, else None.
        """
        p = self.p
        if 'trial_starts_s' not in p or 'trial_duration_s' not in p:
            return None
        starts = np.asarray(p['trial_starts_s'], dtype=float).ravel()
        n = int(starts.size)
        if n == 0:
            return None
        tr = int(trial_index)
        if tr < 0:
            tr += n
        if tr < 0 or tr >= n:
            return None
        t0_trial = float(starts[tr])
        t1_trial = t0_trial + float(p['trial_duration_s'])
        pre_s = max(0.0, float(pre_silence_s))
        post_s = max(0.0, float(post_silence_s))
        t0 = max(0.0, t0_trial - pre_s)
        t1 = min(float(self.duration), t1_trial + post_s)
        created = ax is None
        if created:
            fs = figsize if figsize is not None else (POSTER_COL_INCH, POSTER_COL_INCH)
            fig, ax = plt.subplots(figsize=fs)
        else:
            fig = None
        self.plot_spike_raster(
            ax,
            group_separators=group_separators,
            t_window_s=(t0, t1),
            time_relative_to_window=True,
            relative_time_origin_s=t0_trial,
            title=title,
            scatter_size=scatter_size,
        )
        # Subtle stimulus-period shading to help non-expert readers map protocol timing.
        cs_start = 0.0
        cs_end = cs_start + float(p.get('CS_train_duration', 0 * second) / second)
        us_start = float(p.get('ISI', 0 * second) / second)
        us_end = us_start + float(p.get('US_train_duration', 0 * second) / second)
        # ax.axvspan(cs_start, cs_end, color='#fdbb84', alpha=0.3, lw=0, zorder=0)
        # ax.axvspan(us_start, us_end, color='#9ecae1', alpha=0.3, lw=0, zorder=0)
        if created:
            fig.tight_layout()
        return fig

    def plot_spike_raster_full(
        self,
        ax=None,
        figsize=None,
        group_separators=True,
        title='Spike raster (full simulation)',
        scatter_size=None,
    ):
        """Entire run on one axes; default title 'Spike raster (full simulation)'."""
        created = ax is None
        if created:
            fs = figsize if figsize is not None else (POSTER_COL_INCH, POSTER_COL_INCH)
            fig, ax = plt.subplots(figsize=fs)
        else:
            fig = None
        self.plot_spike_raster(
            ax,
            group_separators=group_separators,
            title=title,
            scatter_size=scatter_size,
        )
        if created:
            fig.tight_layout()
        return fig

    def plot_lowpass_firing_rate_trial(
        self,
        trial_index,
        ax=None,
        figsize=None,
        bin_size=5 * ms,
        tau_s=None,
        title=None,
    ):
        """
        Population mean low-pass firing rate vs time in trial: excitatory CS / US / NS (when defined),
        then inhibitory mean in a second panel. Uses same causal low-pass as PCA (pca_lowpass_tau).

        ax : None or (ax_exc, ax_inh)
            If None, builds a figure with two stacked axes. Otherwise draws on the given pair.
        """
        time_s, rates = compute_single_trial_lowpass_matrix(
            self, trial_index, bin_size=bin_size, tau_s=tau_s
        )
        if time_s is None or rates is None:
            return None
        p = self.p
        n_exc = int(p['nExc'])
        exc = rates[:, :n_exc]
        inh = rates[:, n_exc:]
        created = ax is None
        if created:
            fs = figsize if figsize is not None else (POSTER_COL_INCH, POSTER_COL_INCH)
            fig, (ax_e, ax_i) = plt.subplots(
                2,
                1,
                figsize=fs,
                sharex=True,
                gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12},
            )
        else:
            fig = None
            ax_e, ax_i = ax

        colors_exc = ('C3', 'C0', 'C2')
        if 'cs_neuron_inds' in p and 'us_neuron_inds' in p:
            cs = np.atleast_1d(p['cs_neuron_inds']).astype(int)
            us = np.atleast_1d(p['us_neuron_inds']).astype(int)
            cs_set, us_set = set(cs.tolist()), set(us.tolist())
            ns = np.array([i for i in range(n_exc) if i not in cs_set and i not in us_set], dtype=int)
            if cs.size > 0:
                ax_e.plot(
                    time_s,
                    np.mean(exc[:, cs], axis=1),
                    color=colors_exc[0],
                    lw=POSTER_LINEWIDTH,
                    label='E, CS',
                )
            if us.size > 0:
                ax_e.plot(
                    time_s,
                    np.mean(exc[:, us], axis=1),
                    color=colors_exc[1],
                    lw=POSTER_LINEWIDTH,
                    label='E, US',
                )
            if ns.size > 0:
                ax_e.plot(
                    time_s,
                    np.mean(exc[:, ns], axis=1),
                    color=colors_exc[2],
                    lw=POSTER_LINEWIDTH,
                    label='E, NS',
                )
        else:
            ax_e.plot(
                time_s,
                np.mean(exc, axis=1),
                color='C0',
                lw=POSTER_LINEWIDTH,
                label='E (mean)',
            )
        ax_e.set_ylabel('Exc. rate (Hz)', fontsize=POSTER_LABEL_FS)
        ax_e.tick_params(axis='both', which='major', labelsize=POSTER_TICK_FS)
        ax_e.legend(loc='upper right', fontsize=POSTER_TICK_FS, framealpha=0.92)

        if inh.shape[1] > 0:
            ax_i.plot(
                time_s,
                np.mean(inh, axis=1),
                color='0.25',
                lw=POSTER_LINEWIDTH,
                label='I (mean)',
            )
            ax_i.legend(loc='upper right', fontsize=POSTER_TICK_FS, framealpha=0.92)
        ax_i.set_xlabel('Time in trial (s)', fontsize=POSTER_LABEL_FS)
        ax_i.set_ylabel('Inh. rate (Hz)', fontsize=POSTER_LABEL_FS)
        ax_i.tick_params(axis='both', which='major', labelsize=POSTER_TICK_FS)

        if title is None:
            tr_disp = int(trial_index)
            if 'trial_starts_s' in p:
                starts = np.asarray(p['trial_starts_s'], dtype=float).ravel()
                n_tot = int(starts.size)
                if n_tot > 0:
                    if tr_disp < 0:
                        tr_disp += n_tot
                    tr_disp = max(0, min(tr_disp, n_tot - 1))
                    title = f'Trial {tr_disp + 1} of {n_tot}: low-pass firing rates by group'
            if title is None:
                title = 'Low-pass firing rates by group'
        if created:
            fig.suptitle(title, fontsize=POSTER_TITLE_FS, fontweight='bold', y=0.98)
        else:
            ax_e.set_title(title, fontsize=POSTER_TITLE_FS, fontweight='bold', pad=12)
        if created:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_weight_change_blocks(self, ax=None, figsize=None, title=None):
        """
        Mean ± SEM percentage EE weight change by pre→post block (CS, US, NS).
        Requires weight_matrix_pre/post and cs/us_neuron_inds in params.
        """
        if 'weight_matrix_pre' not in self.p or 'weight_matrix_post' not in self.p:
            return None
        if 'cs_neuron_inds' not in self.p or 'us_neuron_inds' not in self.p:
            return None
        W_pre = np.asarray(self.p['weight_matrix_pre'])
        W_post = np.asarray(self.p['weight_matrix_post'])
        nExc = self.p['nExc']
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
            fs = figsize if figsize is not None else (POSTER_COL_INCH, POSTER_COL_INCH)
            fig, ax = plt.subplots(figsize=fs)
            created_fig = fig
        else:
            created_fig = None
        x = np.arange(len(labels))
        bar_colors = ['C3', 'C0', '0.6']
        colors = [bar_colors[i // 3] if i // 3 < 3 else 'gray' for i in range(len(labels))]
        ax.bar(
            x,
            means,
            yerr=sems,
            capsize=5,
            color=colors,
            alpha=0.9,
            edgecolor='0.2',
            linewidth=0.9,
        )
        ax.axhline(0, color='0.3', linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=POSTER_TICK_FS)
        ax.set_ylabel(r'$\Delta W / W$ (%)', fontsize=POSTER_LABEL_FS)
        ax.tick_params(axis='y', which='major', labelsize=POSTER_TICK_FS)
        if title is None:
            title = r'E$\rightarrow$E weight change by block (mean $\pm$ SEM)'
        ax.set_title(title, fontsize=POSTER_TITLE_FS, fontweight='bold', pad=14, loc='center')
        if created_fig is not None:
            created_fig.tight_layout()
        return created_fig

    def plot_ee_w_in_w_out(self, figsize=None, title=None):
        """
        W_in and W_out vs time (CS, US, NS) with consistent CS/US/NS color key.
        """
        if self.w_stats_t.size == 0 or self.w_in_CS_US_NS.size == 0:
            return None
        fs = figsize if figsize is not None else (POSTER_COL_INCH, POSTER_COL_INCH)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fs, sharex=True)
        t = self.w_stats_t
        # Keep consistent with block-change plots: CS red, US blue, NS gray.
        colors = ('C3', 'C0', '0.45')
        labels = ('CS', 'US', 'NS')
        win = self.w_in_CS_US_NS * 1e9
        wout = self.w_out_CS_US_NS * 1e9
        n_pop = min(3, win.shape[1])
        for k in range(n_pop):
            ax1.plot(
                t,
                np.where(np.isfinite(win[:, k]), win[:, k], np.nan),
                color=colors[k],
                lw=POSTER_LINEWIDTH,
                label=labels[k],
            )
            ax2.plot(
                t,
                np.where(np.isfinite(wout[:, k]), wout[:, k], np.nan),
                color=colors[k],
                lw=POSTER_LINEWIDTH,
                label=labels[k],
            )
        ax1.set_ylabel(r'$W_{\mathrm{in}}$ (nS)', fontsize=POSTER_LABEL_FS)
        ax2.set_ylabel(r'$W_{\mathrm{out}}$ (nS)', fontsize=POSTER_LABEL_FS)
        ax2.set_xlabel('Time (s)', fontsize=POSTER_LABEL_FS)
        for a in (ax1, ax2):
            a.tick_params(axis='both', labelsize=POSTER_TICK_FS)
        handles, legend_labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                legend_labels,
                loc='upper center',
                ncol=min(3, len(handles)),
                bbox_to_anchor=(0.5, 0.89),
                fontsize=max(POSTER_TICK_FS - 2, 8),
                framealpha=0.92,
                borderpad=0.25,
                handlelength=1.6,
                handletextpad=0.4,
                columnspacing=0.9,
                labelspacing=0.25,
            )
        if title is None:
            title = r'Recurrent E weights ($W_{\mathrm{in}}$, $W_{\mathrm{out}}$) by assembly'
        fig.suptitle(title, fontsize=POSTER_TITLE_FS, fontweight='bold', x=0.5, y=0.98, ha='center')
        fig.tight_layout(rect=[0, 0, 1, 1])
        return fig

    def plot_pca_trial_averaged(
        self,
        bin_size=10 * ms,
        n_components=3,
        ax=None,
        use_exc_only=True,
        cmap=None,
        line_alpha=0.95,
        line_lw=None,
        figsize=None,
        title=None,
        pca_view='default',
        pca_view_elev=None,
        pca_view_azim=None,
    ):
        """
        3D trajectory: PCA fit on trial-mean low-pass exc rates; line colored by within-trial time.
        """
        time_s, proj, pca = compute_pca_mean_trajectory_projected(
            self, bin_size, use_exc_only, n_components=n_components
        )
        if proj is None or pca is None or proj.shape[0] < 2:
            return None
        cent = np.asarray(proj, dtype=float)
        if cent.shape[1] < 3:
            cent = np.hstack([cent, np.zeros((cent.shape[0], 3 - cent.shape[1]))])
        else:
            cent = cent[:, :3]

        if cmap is None:
            cmap = POSTER_PCA_CMAP
        if line_lw is None:
            line_lw = POSTER_PCA_LINEWIDTH

        if ax is None:
            fs = figsize if figsize is not None else (POSTER_COL_INCH, POSTER_COL_INCH)
            fig = plt.figure(figsize=fs)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            created_fig = fig
        else:
            created_fig = None

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

        norm = Normalize(vmin=float(time_s.min()), vmax=float(time_s.max()))
        cmap_obj = plt.get_cmap(cmap)
        pc1, pc2, pc3 = cent[:, 0], cent[:, 1], cent[:, 2]
        for i in range(len(time_s) - 1):
            c = cmap_obj(norm((time_s[i] + time_s[i + 1]) / 2))
            ax.plot(
                pc1[i : i + 2],
                pc2[i : i + 2],
                pc3[i : i + 2],
                color=c,
                alpha=line_alpha,
                lw=line_lw,
                solid_capstyle='round',
            )
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        ax.set_xlabel('PC1', fontsize=POSTER_LABEL_FS, labelpad=0)
        ax.set_ylabel('PC2', fontsize=POSTER_LABEL_FS, labelpad=0)
        ax.set_zlabel('PC3', fontsize=POSTER_LABEL_FS, labelpad=0)
        ax.tick_params(axis='x', labelsize=POSTER_TICK_FS, pad=4)
        ax.tick_params(axis='y', labelsize=POSTER_TICK_FS, pad=4)
        ax.tick_params(axis='z', labelsize=POSTER_TICK_FS, pad=4)
        # Keep ticks/grid marks but hide numeric tick labels for a cleaner poster look.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        t_lbl = 'Time in trial (s)' if 'trial_starts_s' in self.p else 'Time (s)'
        cb = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.14)
        cb.set_label(t_lbl, fontsize=POSTER_LABEL_FS)
        cb.ax.tick_params(labelsize=POSTER_TICK_FS)
        # Camera controls:
        # - pca_view='default': use poster-friendly fixed view.
        # - pca_view='auto': search candidate views and pick the most spread/readable.
        # - pca_view='manual': use pca_view_elev / pca_view_azim (falls back to default if omitted).
        if pca_view == 'auto':
            elev, azim = suggest_pca_view_angles(cent)
        elif pca_view == 'manual':
            elev = 22.0 if pca_view_elev is None else float(pca_view_elev)
            azim = 125.0 if pca_view_azim is None else float(pca_view_azim)
        else:
            elev, azim = 22.0, 125.0
        ax.view_init(elev=elev, azim=azim)
        if title is None:
            title = 'Trial-averaged population dynamics (PCA)'
        if created_fig is not None:
            created_fig.suptitle(title, fontsize=POSTER_TITLE_FS, fontweight='bold', x=0.5, y=0.93, ha='center')
            # 3D + colorbar layout is more stable with manual subplot padding than tight_layout.
            created_fig.subplots_adjust(left=0.02, right=0.88, bottom=0.0, top=1)
        else:
            ax.set_title(title, fontsize=POSTER_TITLE_FS, fontweight='bold', pad=12, loc='center')
        return created_fig

    def get_mean_firing_rates(self):
        return compute_mean_firing_rates(self)


def plot_all_figures(
    results,
    show=True,
    save_svg_dir=None,
    pca_view='default',
    pca_view_elev=None,
    pca_view_azim=None,
):
    """
    Poster figures (root simulation defaults): first-trial raster (stimulation protocol),
    EE ΔW, W_in/W_out, PCA.

    Returns
    -------
    tuple
        (fig_raster_first, fig_rates_first, fig_rates_last, fig_weights, fig_wio, fig_pca).
        Firing-rate figure slots are kept as None for backward compatibility.
    """
    fs = (POSTER_COL_INCH, POSTER_COL_INCH)
    fig_raster_first = results.plot_spike_raster_trial(
        0,
        figsize=fs,
        title='Spike raster (first trial)',
    )
    fig_rates_first = None
    fig_rates_last = None

    fig_weights = results.plot_weight_change_blocks(figsize=fs)
    if fig_weights is not None:
        fig_weights.tight_layout()

    fig_wio = results.plot_ee_w_in_w_out(figsize=fs)
    if fig_wio is not None:
        fig_wio.tight_layout()

    fig_pca = results.plot_pca_trial_averaged(
        bin_size=10 * ms,
        n_components=3,
        figsize=fs,
        pca_view=pca_view,
        pca_view_elev=pca_view_elev,
        pca_view_azim=pca_view_azim,
    )

    if save_svg_dir:
        import os
        os.makedirs(save_svg_dir, exist_ok=True)
        figs = [
            ("raster_first_trial", fig_raster_first),
            ("weights_blocks", fig_weights),
            ("weights_in_out", fig_wio),
            ("pca_trial_averaged", fig_pca),
        ]
        for name, fig in figs:
            if fig is not None:
                fig.savefig(os.path.join(save_svg_dir, f"{name}.svg"), format="svg", bbox_inches="tight")

    if show:
        plt.show()
    return fig_raster_first, fig_rates_first, fig_rates_last, fig_weights, fig_wio, fig_pca
