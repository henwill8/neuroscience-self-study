"""
Replicate Litwin-Kumar & Doiron (2014) Figures 2 and 3.
Run simulations first: python run_simulation.py (with_istdp) and python run_simulation.py --no-istdp.
Then: python figures_lk2014.py [--fig2] [--fig3] [--all]
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from analysis import load_results, compute_W_in_W_out

# Paths relative to this script so figures load the same results regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, 'results')
FIG_DIR = os.path.join(_SCRIPT_DIR, 'figures')


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _assembly_order(patterns, n_patterns_use=3):
    """Return permutation of neuron indices that groups by assembly (for raster sort)."""
    N_E = patterns.shape[1]
    # Assign each neuron to its "primary" assembly (first pattern it belongs to)
    primary = np.full(N_E, -1)
    for k in range(patterns.shape[0]):
        for i in np.where(patterns[k, :])[0]:
            if primary[i] < 0:
                primary[i] = k
    # Order: assembly 0, then 1, then 2, then unassigned
    order = []
    for k in range(min(n_patterns_use, patterns.shape[0])):
        order.extend(np.where(primary == k)[0].tolist())
    rest = np.where(primary < 0)[0]
    order.extend(rest.tolist())
    remaining = [i for i in range(N_E) if i not in order]
    order.extend(remaining)
    return np.array(order), primary


# ---------- Figure 2 ----------

def plot_fig2b(results_path=None, n_assemblies_show=3, save=True):
    """Fig 2b: (b) W_in for each assembly (top, one line per assembly) and W_out (bottom).
    n_assemblies_show: number of assemblies to plot (default 3); None = all."""
    _ensure_fig_dir()
    res = load_results(results_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    t = np.asarray(res['W_in_t'])
    t_train_end = res.get('t_training_end', t[-1] * 0.8)
    if len(t) > 0 and t.max() > 0:
        t_warmup = res.get('t_warmup_s')
        phase2_total = res.get('phase2_total_s')
        if t_warmup is not None and phase2_total is not None:
            expected_end = t_warmup + phase2_total
            if abs(t_train_end - expected_end) > 1.0 and t_train_end < 0.2 * t.max():
                t_train_end = expected_end

    # W_in per assembly: one line per assembly (within-assembly average)
    W_in_per_assembly = res.get('W_in_per_assembly_vals')
    if W_in_per_assembly is None or len(W_in_per_assembly) == 0:
        raise ValueError('Result must contain W_in_per_assembly_vals')
    W_in_per_assembly = np.array(W_in_per_assembly, dtype=float)
    n_assemblies = W_in_per_assembly.shape[1] if W_in_per_assembly.ndim >= 2 else 1
    if n_assemblies_show is not None:
        n_assemblies = min(n_assemblies_show, n_assemblies)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 4), sharex=True, height_ratios=[1, 1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_assemblies, 1)))
    for k in range(n_assemblies):
        w = W_in_per_assembly[:, k] * 1e9
        if np.any(np.isfinite(w)):
            ax_top.plot(t, np.where(np.isfinite(w), w, np.nan), color=colors[k % len(colors)],
                        lw=1.2, alpha=0.9, label='Assembly %d' % (k + 1))
    ax_top.axvline(t_train_end, color='gray', ls='--', alpha=0.8, label='Training end')
    ax_top.set_ylabel('Synaptic strength (nS)')
    ax_top.set_title(r'$W_{\mathrm{in}}$ (within assembly, one line per assembly)')
    ax_top.legend(loc='upper right', fontsize=7, ncol=2)
    ax_top.set_xlim(left=0)

    # W_out per assembly: one line per assembly (between that assembly and others)
    W_out_per_assembly = res.get('W_out_per_assembly_vals')
    if W_out_per_assembly is None or len(W_out_per_assembly) == 0:
        raise ValueError('Result must contain W_out_per_assembly_vals')
    W_out_per_assembly = np.array(W_out_per_assembly, dtype=float)
    n_out = W_out_per_assembly.shape[1] if W_out_per_assembly.ndim >= 2 else 1
    if n_assemblies_show is not None:
        n_out = min(n_assemblies_show, n_out)
    for k in range(n_out):
        w = W_out_per_assembly[:, k] * 1e9
        if np.any(np.isfinite(w)):
            ax_bot.plot(t, np.where(np.isfinite(w), w, np.nan), color=colors[k % len(colors)],
                        lw=1.2, alpha=0.9, label='Assembly %d' % (k + 1))
    ax_bot.axvline(t_train_end, color='gray', ls='--', alpha=0.8, label='Training end')
    ax_bot.set_xlabel('Time (s)')
    ax_bot.set_ylabel('Synaptic strength (nS)')
    ax_bot.set_title(r'$W_{\mathrm{out}}$ (between assemblies, one line per assembly)')
    ax_bot.legend(loc='upper right', fontsize=7, ncol=2)
    ax_bot.set_xlim(left=0)
    fig.suptitle('Fig 2b: Average synaptic weight within vs between assemblies', fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'fig2b_Win_Wout.pdf'), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(FIG_DIR, 'fig2b_Win_Wout.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def _draw_connectivity_graph(ax, g_EE, i_ee, j_ee, subset, subset_set, n_neurons, n_assemblies,
                            stimulus1_neurons, title):
    """Draw one graph: orange = strong connections, nodes coloured by stimulus 1 (pattern 0)."""
    g = np.asarray(g_EE)
    if hasattr(g.flat[0], 'item'):
        g = np.array([float(x) for x in g])
    thresh = np.median(g)
    edges_strong, edges_weak = [], []
    for idx in range(len(g)):
        pre, post = int(i_ee[idx]), int(j_ee[idx])
        if pre not in subset_set or post not in subset_set:
            continue
        w = float(g[idx])
        sub_i = np.where(subset == pre)[0][0]
        sub_j = np.where(subset == post)[0][0]
        if w >= thresh:
            edges_strong.append((sub_i, sub_j))
        else:
            edges_weak.append((sub_i, sub_j))
    pos = {}
    n_per = max(1, n_neurons // n_assemblies)
    for a in range(n_assemblies):
        for i in range(n_per):
            idx = a * n_per + i
            if idx >= n_neurons:
                break
            pos[idx] = (a * 1.5 + 0.2 * (i % 5), 0.2 * (i // 5) + a * 3)
    for idx in range(n_neurons):
        if idx not in pos:
            pos[idx] = (idx % 8, idx // 8)
    for (i, j) in edges_weak:
        if i in pos and j in pos:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'gray', alpha=0.25, lw=0.5)
    for (i, j) in edges_strong:
        if i in pos and j in pos:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'orange', alpha=0.8, lw=1)
    node_colors = ['C0' if subset[i] in stimulus1_neurons else 'lightgray' for i in range(n_neurons)]
    ax.scatter([pos[i][0] for i in range(n_neurons)], [pos[i][1] for i in range(n_neurons)],
               c=node_colors, s=20, edgecolors='k', linewidths=0.5)
    ax.set_title(title)
    ax.axis('off')


def plot_fig2c(results_path=None, n_neurons=50, n_assemblies=3, save=True):
    """Fig 2c: Connection strength for 50 neurons from 3 assemblies before (left) and after (right)
    training. Orange = strong excitatory connections. Coloured nodes = neurons targeted by stimulus 1."""
    _ensure_fig_dir()
    res = load_results(results_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    patterns = res['patterns']
    i_ee = np.asarray(res['i_ee'], dtype=int)
    j_ee = np.asarray(res['j_ee'], dtype=int)
    order, _ = _assembly_order(patterns, n_assemblies)
    subset = order[:n_neurons]
    subset_set = set(subset)
    stimulus1_neurons = set(np.where(patterns[0, :])[0])  # neurons targeted by stimulus 1 (pattern 0)

    g_EE_after = np.asarray(res['g_EE'])
    g_EE_before = np.asarray(res['g_EE_start'])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
    _draw_connectivity_graph(ax_left, g_EE_before, i_ee, j_ee, subset, subset_set,
                             n_neurons, n_assemblies, stimulus1_neurons, 'Before training')
    _draw_connectivity_graph(ax_right, g_EE_after, i_ee, j_ee, subset, subset_set,
                             n_neurons, n_assemblies, stimulus1_neurons, 'After training')
    fig.suptitle('Fig 2c: Connection strength (orange = strong). Coloured = stimulus 1.', fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'fig2c_connectivity.pdf'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def _raster_rows_by_assembly_with_repeats(patterns, n_assemblies_use=3):
    """Row order: for each assembly in order, list all neurons in that assembly (neurons can repeat).
    Only include neurons that belong to at least one assembly. Returns (row_order, neuron_to_row)."""
    in_any = np.any(patterns[:n_assemblies_use, :], axis=0)
    neurons_in_any = np.where(in_any)[0]
    row_order = []  # list of neuron indices, one per row (repeated if in multiple assemblies)
    for k in range(n_assemblies_use):
        in_k = np.where(patterns[k, :])[0]
        row_order.extend(in_k.tolist())
    return row_order, neurons_in_any


def plot_fig2d_2e(results_path=None, duration=2.0, n_assemblies_use=3, save=True):
    """Fig 2d: Excitatory spike rasters before and after training. Adjacent rows = same assembly;
    rows repeated for multi-assembly neurons. Only assembly neurons shown.
    Fig 2e: Average firing rate for each assembly, corresponding to activity in d."""
    _ensure_fig_dir()
    res = load_results(results_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    t_warmup = res.get('t_warmup_s', 0)
    t_train_end = res.get('t_training_end', 0)
    sp_t = np.asarray(res['sp_E_t'])
    sp_i = np.asarray(res['sp_E_i'])
    patterns = res['patterns']
    N_E = res['params']['N_E']

    row_order, neurons_in_any = _raster_rows_by_assembly_with_repeats(patterns, n_assemblies_use)
    n_rows = len(row_order)
    neuron_to_rows = {}  # for each neuron, list of row indices where it appears
    for r, neur in enumerate(row_order):
        neuron_to_rows.setdefault(neur, []).append(r)

    # Before: end of warmup; After: start of spontaneous
    t_before = (max(0, t_warmup - duration), t_warmup)
    t_after = (t_train_end, t_train_end + duration)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='col')
    ax_d_left, ax_d_right = axes[0, 0], axes[0, 1]
    for ax, (t0, t1), title in [(ax_d_left, t_before, '(d) Before training'), (ax_d_right, t_after, '(d) After training')]:
        mask = (sp_t >= t0) & (sp_t < t1)
        t_sub = sp_t[mask]
        i_sub = sp_i[mask]
        # Each spike (t, neuron): if neuron in an assembly, plot at (t, r) for each row r that neuron occupies
        t_plot = []
        row_plot = []
        for ti, ni in zip(t_sub, i_sub):
            if ni in neuron_to_rows:
                for r in neuron_to_rows[ni]:
                    t_plot.append(ti)
                    row_plot.append(r)
        if t_plot:
            ax.scatter(t_plot, row_plot, s=0.2, c='k', alpha=0.7)
        ax.set_ylabel('Neuron row (by assembly)')
        ax.set_title(title)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.set_xlim(t0, t1)

    # (e) Average firing rate for neurons in each assembly (corresponding to activity in d)
    ax_e_left, ax_e_right = axes[1, 0], axes[1, 1]
    bin_w = 0.1
    for ax, (t0, t1), title in [(ax_e_left, t_before, '(e) Before training'), (ax_e_right, t_after, '(e) After training')]:
        bins = np.arange(t0, t1 + bin_w, bin_w)
        n_asm = min(n_assemblies_use, patterns.shape[0])
        colors = plt.cm.tab10(np.linspace(0, 1, n_asm))
        for k in range(n_asm):
            in_k = np.where(patterns[k, :])[0]
            mask_k = (sp_t >= t0) & (sp_t < t1) & np.isin(sp_i, in_k)
            t_k = sp_t[mask_k]
            n_k = len(in_k)
            rate_k = np.histogram(t_k, bins=bins)[0] / (n_k * bin_w) if n_k > 0 else np.zeros(len(bins) - 1)
            ax.plot(bins[:-1], rate_k, color=colors[k], lw=1.5, label='Assembly %d' % (k + 1))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (Hz)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(t0, t1)
    fig.suptitle('Fig 2d–e: Rasters (assembly-ordered, rows repeated) and rate per assembly', fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'fig2d_2e_raster_rate.pdf'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ---------- Figure 3 ----------

def plot_fig3b(results_with_path=None, results_without_path=None, save=True):
    """Fig 3b: Average inhibitory connection strength vs assembly size (with vs without iSTDP)."""
    _ensure_fig_dir()
    res_w = load_results(results_with_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    res_wo = load_results(results_without_path or os.path.join(RESULTS_DIR, 'assembly_no_istdp.pkl'))
    if 'j_ei' not in res_w or res_w.get('g_EI') is None:
        print('Fig 3b: j_ei or g_EI missing in results; re-run simulation to save EI connectivity.')
        return None
    patterns = res_w['patterns']
    g_EI_w = np.asarray(res_w['g_EI'])
    g_EI_wo = np.asarray(res_wo['g_EI'])
    j_ei = np.asarray(res_w['j_ei'], dtype=int)
    if hasattr(g_EI_w[0], 'item'):
        g_EI_w = np.array([float(x) for x in g_EI_w])
        g_EI_wo = np.array([float(x) for x in g_EI_wo])
    sizes = []
    mean_EI_w = []
    mean_EI_wo = []
    for k in range(patterns.shape[0]):
        in_k = np.where(patterns[k, :])[0]
        if len(in_k) < 2:
            continue
        in_set = set(in_k)
        vals_w = [g_EI_w[idx] for idx in range(len(g_EI_w)) if j_ei[idx] in in_set]
        vals_wo = [g_EI_wo[idx] for idx in range(len(g_EI_wo)) if j_ei[idx] in in_set]
        if vals_w and vals_wo:
            sizes.append(len(in_k))
            mean_EI_w.append(np.mean(vals_w) * 1e9)
            mean_EI_wo.append(np.mean(vals_wo) * 1e9)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(sizes, mean_EI_w, c='green', label='With iSTDP', s=40, alpha=0.8)
    ax.scatter(sizes, mean_EI_wo, c='red', label='Without iSTDP', s=40, alpha=0.8)
    ax.set_xlabel('Assembly size')
    ax.set_ylabel('Average inhibitory connection strength (nS)')
    ax.legend()
    ax.set_title('Fig 3b: Average inhibitory connection strength onto assemblies of different sizes')
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'fig3b_JEI_vs_assembly_size.pdf'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_fig3c(results_with_path=None, results_without_path=None, save=True):
    """Fig 3c: Average W_in within assemblies during training: with iSTDP (left) and without (right)."""
    _ensure_fig_dir()
    res_w = load_results(results_with_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    res_wo = load_results(results_without_path or os.path.join(RESULTS_DIR, 'assembly_no_istdp.pkl'))
    t_train_end = res_w.get('t_training_end', 0)
    # Single W_in curve = mean across assemblies (from per-assembly data)
    win_w = np.array(res_w['W_in_per_assembly_vals'], dtype=float)
    win_wo = np.array(res_wo['W_in_per_assembly_vals'], dtype=float)
    ww = np.nanmean(win_w, axis=1) * 1e9
    wwo = np.nanmean(win_wo, axis=1) * 1e9
    tw = np.asarray(res_w['W_in_t'])
    two = np.asarray(res_wo['W_in_t'])
    mask_w = tw <= t_train_end
    mask_wo = two <= t_train_end
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    ax_left.plot(tw[mask_w], ww[mask_w], 'green', lw=1.5)
    ax_left.set_xlabel('Time (s)')
    ax_left.set_ylabel(r'$W_{\mathrm{in}}$ (nS)')
    ax_left.set_title('With iSTDP')
    ax_left.set_xlim(left=0)
    ax_right.plot(two[mask_wo], wwo[mask_wo], 'red', lw=1.5)
    ax_right.set_xlabel('Time (s)')
    ax_right.set_title('Without iSTDP')
    ax_right.set_xlim(left=0)
    fig.suptitle('Fig 3c: Average connection strength within assemblies during training', fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'fig3c_Win_training.pdf'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_fig3d(results_with_path=None, results_without_path=None, duration=2.0, save=True):
    """Fig 3d: Spike rasters during spontaneous activity: with iSTDP (left) and without (right)."""
    _ensure_fig_dir()
    res_w = load_results(results_with_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    res_wo = load_results(results_without_path or os.path.join(RESULTS_DIR, 'assembly_no_istdp.pkl'))
    t_train_end = res_w.get('t_training_end', 0)
    t0 = t_train_end
    t1 = t0 + duration
    patterns = res_w['patterns']
    N_E = res_w['params']['N_E']
    order, _ = _assembly_order(patterns, 3)
    inv_order = np.empty(N_E, dtype=int)
    inv_order[order] = np.arange(N_E)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    for ax, res, title in [(ax_left, res_w, 'With iSTDP'), (ax_right, res_wo, 'Without iSTDP')]:
        sp_t = np.asarray(res['sp_E_t'])
        sp_i = np.asarray(res['sp_E_i'])
        mask = (sp_t >= t0) & (sp_t < t1)
        t_sub = sp_t[mask]
        row = inv_order[sp_i[mask]]
        ax.scatter(t_sub, row, s=0.3, c='k', alpha=0.7)
        ax.set_ylabel('Neuron')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.set_ylim(-0.5, N_E - 0.5)
    fig.suptitle('Fig 3d: Spontaneous activity (left: with iSTDP, right: without iSTDP)', fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'fig3d_raster_compare.pdf'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_full_raster_single(results_path=None, save=True, title='Full raster (training + spontaneous)'):
    """Full raster for a single result: warmup + training + spontaneous, neurons ordered by assembly."""
    _ensure_fig_dir()
    res = load_results(results_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    t_train_end = res.get('t_training_end', 0)
    patterns = res['patterns']
    N_E = res['params']['N_E']
    order, _ = _assembly_order(patterns, 3)
    inv_order = np.empty(N_E, dtype=int)
    inv_order[order] = np.arange(N_E)
    sp_t = np.asarray(res['sp_E_t'])
    sp_i = np.asarray(res['sp_E_i'])
    row = inv_order[sp_i]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.scatter(sp_t, row, s=0.15, c='k', alpha=0.6)
    ax.axvline(t_train_end, color='red', ls='--', alpha=0.8, label='Training end')
    ax.set_ylabel('Neuron')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim(-0.5, N_E - 0.5)
    ax.set_xlim(left=0)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'experiment_istdp_raster.pdf'), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(FIG_DIR, 'experiment_istdp_raster.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_full_raster_compare(results_with_path=None, results_without_path=None, save=True):
    """Full raster: warmup + training + spontaneous for with iSTDP (left) and without iSTDP (right)."""
    _ensure_fig_dir()
    res_w = load_results(results_with_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    res_wo = load_results(results_without_path or os.path.join(RESULTS_DIR, 'assembly_no_istdp.pkl'))
    t_train_end = res_w.get('t_training_end', 0)
    patterns = res_w['patterns']
    N_E = res_w['params']['N_E']
    order, _ = _assembly_order(patterns, 3)
    inv_order = np.empty(N_E, dtype=int)
    inv_order[order] = np.arange(N_E)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
    for ax, res, title in [(ax_left, res_w, 'With iSTDP'), (ax_right, res_wo, 'Without iSTDP')]:
        sp_t = np.asarray(res['sp_E_t'])
        sp_i = np.asarray(res['sp_E_i'])
        row = inv_order[sp_i]
        ax.scatter(sp_t, row, s=0.15, c='k', alpha=0.6)
        ax.axvline(t_train_end, color='red', ls='--', alpha=0.8, label='Training end')
        ax.set_ylabel('Neuron')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.set_ylim(-0.5, N_E - 0.5)
        ax.set_xlim(left=0)
    fig.suptitle('Full raster: training + spontaneous (left: with iSTDP, right: without iSTDP)', fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, 'full_raster_compare.pdf'), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(FIG_DIR, 'full_raster_compare.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def print_mean_firing_rate(results_path=None):
    """Report mean firing rate (should be ~1.7 Hz after training)."""
    res = load_results(results_path or os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl'))
    t_end = res['sp_E_t'].max() if len(res['sp_E_t']) else 0
    t_train = res.get('t_training_end', t_end * 0.8)
    # Spontaneous: after t_train
    mask = res['sp_E_t'] >= t_train
    n_spikes = np.sum(mask)
    duration_s = t_end - t_train
    N_E = res['params']['N_E']
    if duration_s > 0:
        rate = n_spikes / (N_E * duration_s)
        print('Mean firing rate (spontaneous): %.3f Hz' % rate)
    return rate if duration_s > 0 else None


# ---------- Main ----------

if __name__ == '__main__':
    import sys
    _ensure_fig_dir()
    do_fig2 = '--fig2' in sys.argv or '--all' in sys.argv or len(sys.argv) == 1
    do_fig3 = '--fig3' in sys.argv or '--all' in sys.argv
    do_full_raster = '--full-raster' in sys.argv

    if do_full_raster:
        if os.path.isfile(os.path.join(RESULTS_DIR, 'assembly_no_istdp.pkl')):
            plot_full_raster_compare()
        else:
            print('Run with --no-istdp first to generate assembly_no_istdp.pkl.')
    if do_fig2:
        print('Plotting Figure 2...')
        plot_fig2b()
        plot_fig2c()
        plot_fig2d_2e()
    if do_fig3:
        print('Plotting Figure 3...')
        if os.path.isfile(os.path.join(RESULTS_DIR, 'assembly_no_istdp.pkl')):
            plot_fig3b()
            plot_fig3c()
            plot_fig3d()
            plot_full_raster_compare()
        else:
            print('Run with --no-istdp first to generate assembly_no_istdp.pkl for Fig 3.')
    print_mean_firing_rate()
    print('Figures saved to %s/' % FIG_DIR)
