import numpy as np
from scipy.special import gammaln
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Black (zero weight) to yellow (max weight) for weight matrices
BLACK_YELLOW_CMAP = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'], N=256)


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
    Mean correlation within (CS, US, other) and between (CS-US, CS-other, US-other).
    Returns dict or None if CS/US not in params.
    """
    if 'cs_neuron_inds' not in results.p or 'us_neuron_inds' not in results.p:
        return None
    X = compute_population_matrix(results, bin_size, use_exc_only=True, subtract_mean=False)
    R = np.corrcoef(X.T)
    nExc = results.p['nExc']
    cs_set = set(results.p['cs_neuron_inds'])
    us_set = set(results.p['us_neuron_inds'])
    other_inds = np.array([i for i in range(nExc) if i not in cs_set and i not in us_set])
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
    out['within_other'] = mean_upper_triangle(R[np.ix_(other_inds, other_inds)]) if len(other_inds) >= 2 else np.nan
    out['between_CS_US'] = np.nanmean(R[np.ix_(cs_inds, us_inds)]) if len(cs_inds) and len(us_inds) else np.nan
    out['between_CS_other'] = np.nanmean(R[np.ix_(cs_inds, other_inds)]) if len(cs_inds) and len(other_inds) else np.nan
    out['between_US_other'] = np.nanmean(R[np.ix_(us_inds, other_inds)]) if len(us_inds) and len(other_inds) else np.nan
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
        X = X[upstate_mask] - X[upstate_mask].mean(axis=0)
    else:
        X = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = (S ** 2) / (S ** 2).sum()
    return var_explained * 100


def compute_block_weight_change(W_pre, W_post, groups, group_names=None):
    """
    Block-averaged weight change: for each (post, pre) group pair, compute mean and SEM
    of percentage change (W_post - W_pre) / W_pre * 100 over synapses with W_pre > 0.

    groups: 1d array of length N, group index per neuron (0=CS, 1=US, 2=other, etc.)
    group_names: optional list of names for labels (e.g. ['CS', 'US', 'other'])
    Returns: labels (e.g. ['CS→CS', 'CS→US', ...]), means (%), sems (%).
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
            labels.append(f"{group_names[i_post]}→{group_names[i_pre]}")
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
        self.stateMonExcV = stateMonExc.v / mV
        self.stateMonInhV = stateMonInh.v / mV
        self.stateDT = float(stateMonExc.clock.dt / second)
        self.duration = float(params['duration'] / second)
        self.stateMonT = np.arange(0, self.duration, self.stateDT)

    def plot_spike_raster(self, ax):
        nExc = self.p['nExc']
        nInh = self.p['nInh']
        if 'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p:
            cs_set = set(self.p['cs_neuron_inds'])
            us_set = set(self.p['us_neuron_inds'])
            cs_sorted = np.sort(self.p['cs_neuron_inds'])
            us_sorted = np.sort(self.p['us_neuron_inds'])
            other_sorted = np.sort([i for i in range(nExc) if i not in cs_set and i not in us_set])
            order = np.concatenate([cs_sorted, us_sorted, other_sorted])
            neuron_to_display = {n: i for i, n in enumerate(order)}
            nCS, nUS = len(cs_sorted), len(us_sorted)
            t_exc = np.asarray(self.spikeMonExcT)
            i_exc = np.asarray(self.spikeMonExcI)
            y_exc = np.array([neuron_to_display[i] for i in i_exc])
            ax.scatter(t_exc, y_exc, c='blue', s=1, marker='.', linewidths=0)
            ax.axhline(nCS - 0.5, color='k', lw=0.5, linestyle='-')
            ax.axhline(nCS + nUS - 0.5, color='k', lw=0.5, linestyle='-')
            ax.scatter(self.spikeMonInhT, nExc + self.spikeMonInhI, s=1, c='red', marker='.', linewidths=0)
            ax.axhline(nExc - 0.5, color='k', lw=0.5, linestyle='-')
            ax.set_ylim(-0.5, self.p['nUnits'] - 0.5)
            ax.set_ylabel("Neuron (CS | US | other | inh)")
        else:
            ax.scatter(self.spikeMonExcT, self.spikeMonExcI, s=1, c='cyan', marker='.')
            ax.scatter(self.spikeMonInhT, nExc + self.spikeMonInhI, s=1, c='red', marker='.', linewidths=0)
            ax.set_ylim(-0.5, self.p['nUnits'] - 0.5)
            ax.set_ylabel("Neuron index")
        ax.set_xlim(0, self.duration)
        ax.set_xlabel("Time (s)")

    def plot_firing_rate(self, ax, bin_size=5*ms, show_upstate=True,
                         upstate_bin_size=10*ms, p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0):
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        nExc = self.p['nExc']
        nInh = self.p['nInh']
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
            nOther = nExc - nCS - nUS
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
            if nOther > 0:
                other_set = set(range(nExc)) - cs_set - us_set
                mask_other = np.isin(self.spikeMonExcI, np.array(list(other_set)))
                t_other = self.spikeMonExcT[mask_other]
                FR_other, _ = np.histogram(t_other, bins)
                FR_other = FR_other / bin_size_s / nOther
                ax.plot(centers, FR_other, color='0.6', alpha=0.6, label='other exc')
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

        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        if (show_upstate and drew_upstate_span) or ('cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p):
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
        """Plot mean voltage ± error (SEM or SD) for each group (CS, US, other exc, inh) on one axes."""
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

        def plot_group(neuron_inds, V, record_inds, color, label):
            # Map neuron indices to row indices: row k in V corresponds to neuron record_inds[k]
            neuron_inds = np.atleast_1d(neuron_inds)
            row_inds = np.where(np.isin(record_inds, neuron_inds))[0]
            if len(row_inds) == 0:
                return
            sub = V[row_inds]
            mean = sub.mean(axis=0)
            n = sub.shape[0]
            err = sub.std(axis=0) / (np.sqrt(n) if use_sem else 1.0) if n > 1 else np.zeros_like(mean)
            ax.plot(t, mean, color=color, lw=0.8, label=label)
            ax.fill_between(t, mean - err, mean + err, color=color, alpha=0.3)

        if 'cs_neuron_inds' in self.p and 'us_neuron_inds' in self.p:
            cs_inds = np.atleast_1d(self.p['cs_neuron_inds'])
            us_inds = np.atleast_1d(self.p['us_neuron_inds'])
            nExc = self.p['nExc']
            other_inds = np.array([i for i in range(nExc) if i not in cs_inds and i not in us_inds])
            plot_group(cs_inds, V_exc, rec_exc, 'C3', 'CS')
            plot_group(us_inds, V_exc, rec_exc, 'C0', 'US')
            plot_group(other_inds, V_exc, rec_exc, '0.6', 'other exc')
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
        labels = ['within CS', 'within US', 'within other', 'between CS–US', 'between CS–other', 'between US–other']
        values = [cor['within_CS'], cor['within_US'], cor['within_other'],
                  cor['between_CS_US'], cor['between_CS_other'], cor['between_US_other']]
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

    def plot_weight_change_blocks(self, ax=None):
        """
        Bar graph of mean ± SEM percentage weight change by block (CS→CS, CS→US, US→CS, US→US, etc.)
        Uses EE block only with groups CS, US, other. Requires weight_matrix_pre/post and cs/us_neuron_inds.
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
        other_inds = np.array([i for i in range(nExc) if i not in cs_inds and i not in us_inds])
        groups = np.zeros(nExc, dtype=int)
        groups[cs_inds] = 0
        groups[us_inds] = 1
        groups[other_inds] = 2
        group_names = ['CS', 'US', 'other']
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

