import numpy as np
from scipy.special import gammaln
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


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
    # log P(y|s) = y*log(lam_s) - lam_s - gammaln(y+1)
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


class SimpleResults:
    def __init__(self, spikeMonExc, spikeMonInh,
                 stateMonExc, stateMonInh,
                 params):

        self.p = params

        # --- Store spike data (convert to seconds)
        self.spikeMonExcT = spikeMonExc.t / second
        self.spikeMonExcI = spikeMonExc.i

        self.spikeMonInhT = spikeMonInh.t / second
        self.spikeMonInhI = spikeMonInh.i

        # --- Store voltages (convert to mV)
        self.stateMonExcV = stateMonExc.v / mV
        self.stateMonInhV = stateMonInh.v / mV

        # --- Time vector for voltage traces
        self.stateDT = float(stateMonExc.clock.dt / second)
        self.duration = float(params['duration'] / second)
        self.stateMonT = np.arange(0, self.duration, self.stateDT)

    # -------------------------------------------------
    # Build population activity matrix (binned rates)
    # -------------------------------------------------
    def compute_population_matrix(self, bin_size=5*ms, use_exc_only=True, subtract_mean=True):

        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)

        n_neurons = self.p['nExc'] if use_exc_only else self.p['nUnits']
        X = np.zeros((len(bins) - 1, n_neurons))

        for i in range(self.p['nExc']):
            spikes = self.spikeMonExcT[self.spikeMonExcI == i]
            counts, _ = np.histogram(spikes, bins)
            X[:, i] = counts / bin_size_s  # Hz

        if not use_exc_only:
            for i in range(self.p['nInh']):
                spikes = self.spikeMonInhT[self.spikeMonInhI == i]
                counts, _ = np.histogram(spikes, bins)
                X[:, self.p['nExc'] + i] = counts / bin_size_s

        if subtract_mean:
            X = X - X.mean(axis=0)
        return X

    # -------------------------------------------------
    # PCA projection onto first n components (for 3D plot)
    # -------------------------------------------------
    def compute_pca_projection(self, bin_size=5*ms, use_exc_only=True, n_components=3):
        """
        Project binned population activity onto the first n_components PCs.
        Returns time_centers (s), and proj (n_bins, n_components) in same order as PCs.
        """
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2

        X = self.compute_population_matrix(bin_size, use_exc_only)
        # SVD: X = U @ diag(S) @ Vt; projection onto first k right singular vectors is X @ Vt[:k].T
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ Vt[:n_components].T  # (n_bins, n_components)
        return centers, proj

    # -------------------------------------------------
    # Upstate detection (HMM on population spike count, Luczak/Chen et al. style)
    # -------------------------------------------------
    def detect_upstates(self, bin_size=10*ms, use_exc_only=True,
                        p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0):
        """
        Label each time bin as UP or DOWN using a two-state hidden Markov model
        on the population spike count (pooled excitatory activity), as in
        Luczak et al. / Saleem et al. / Chen et al.: the population spike count
        is treated as a single point process whose rate is modulated by a
        discrete hidden state. Viterbi decoding gives the state per bin.
        Adjacent bins in the same state define UP and DOWN periods.

        Parameters (Chen et al. 2009 style):
        - bin_size: bin T for population count (default 10 ms).
        - p_stay: P(same state next bin); 1 - p_stay = P(switch). Paper: PDD=PUU=0.9.
        - rate_up_ratio, rate_down_ratio: relative Poisson rates (α=3, α+μ=1 in paper).
          Actual rates are scaled so (λ_down + λ_up)/2 = observed mean count.

        Returns time_centers (s), upstate_mask (bool, length n_bins).
        """
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        n_bins = len(centers)

        # Population spike count per bin (pool excitatory spikes)
        count, _ = np.histogram(self.spikeMonExcT, bins)
        count = np.asarray(count, dtype=float)

        # Scale emission rates to data: keep paper ratio (UP/DOWN = 3), mean = observed mean
        mean_count = np.maximum(count.mean(), 1.0)
        r = rate_up_ratio / rate_down_ratio  # λ_up / λ_down
        lambda_down = (2 * mean_count) / (1 + r)
        lambda_up = lambda_down * r

        # Transition matrix: row i = current state, col j = next state; 0=DOWN, 1=UP
        p_switch = 1.0 - p_stay
        trans_mat = np.array([[p_stay, p_switch], [p_switch, p_stay]])

        states = _hmm_viterbi_poisson(count, trans_mat, lambda_down, lambda_up)
        upstate_mask = (states == 1)
        return centers, upstate_mask

    # -------------------------------------------------
    # Plot first 3 PCs as connected line; upstate segments colored differently
    # -------------------------------------------------
    def plot_pca_3d_time_color(self, ax=None, bin_size=5*ms, use_exc_only=True,
                               use_upstate_only=True, upstate_bin_size=10*ms,
                               p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0,
                               line_alpha=0.85, line_lw=0.8, cmap='viridis'):
        """
        Plot the first 3 PCs of population activity. Color = time during trial.

        use_upstate_only: if True, PCA is fit on upstate bins only and the trajectory
          shows only upstate times. If False, PCA is fit on the full simulation and
          the full trajectory is plotted.
        """
        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2

        X = self.compute_population_matrix(bin_size, use_exc_only, subtract_mean=False)

        if use_upstate_only:
            _, upstate_mask_coarse = self.detect_upstates(
                bin_size=upstate_bin_size, use_exc_only=use_exc_only,
                p_stay=p_stay, rate_up_ratio=rate_up_ratio, rate_down_ratio=rate_down_ratio
            )
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

    # -------------------------------------------------
    # Raster plot (excitatory with CS/US/other grouping, then inhibitory combined below)
    # -------------------------------------------------
    def plot_spike_raster(self, ax):
        nExc = self.p['nExc']
        nInh = self.p['nInh']

        # Excitatory: if CS/US defined, order as [CS | US | other] and color by group
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
            colors_exc = np.array(['C3' if i in cs_set else ('C0' if i in us_set else '0.6') for i in i_exc])

            ax.scatter(t_exc, y_exc, c=colors_exc, s=1, marker='.', linewidths=0)
            ax.axhline(nCS - 0.5, color='k', lw=0.5, linestyle='-')
            ax.axhline(nCS + nUS - 0.5, color='k', lw=0.5, linestyle='-')
            # Inhibitory combined below excitatory
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

    # -------------------------------------------------
    # Firing rate
    # -------------------------------------------------
    def plot_firing_rate(self, ax, bin_size=5*ms, show_upstate=True,
                         upstate_bin_size=10*ms, p_stay=0.9, rate_up_ratio=3.0, rate_down_ratio=1.0):

        bin_size_s = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size_s)
        centers = bins[:-1] + bin_size_s / 2
        nExc = self.p['nExc']
        nInh = self.p['nInh']

        drew_upstate_span = False
        if show_upstate:
            _, upstate_mask = self.detect_upstates(
                bin_size=upstate_bin_size, use_exc_only=True,
                p_stay=p_stay, rate_up_ratio=rate_up_ratio, rate_down_ratio=rate_down_ratio
            )
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

        # Firing rate by group: CS, US, other exc, inh (if CS/US defined); else exc + inh
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

    # -------------------------------------------------
    # Voltage trace
    # -------------------------------------------------
    def plot_voltage(self, ax, unitType='Exc', neuron_index=0, mean=False):
        """
        Plot voltage trace. If mean=True, plot mean voltage over all recorded neurons
        (ignores neuron_index). Otherwise plot the trace for the given neuron_index.
        """
        if unitType == 'Exc':
            V = np.asarray(self.stateMonExcV)
            color = 'cyan'
        else:
            V = np.asarray(self.stateMonInhV)
            color = 'red'

        if mean:
            v = V.mean(axis=0)
            lw = 0.8
        else:
            v = V[neuron_index]
            lw = 0.6

        ax.plot(self.stateMonT, v, color=color, lw=lw)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
