"""
Analysis: W_in vs W_out, Fano factor, spike count correlations.
Expects results from run_simulation (pickle with g_EE, patterns, sp_E_*, etc.).
"""
import numpy as np
import pickle
import os


def load_results(path='results/assembly_simulation_results.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_W_in_W_out(g_EE, i_ee, j_ee, patterns):
    """
    Given current g_EE (array), connectivity (i_ee, j_ee), and patterns (n_patterns x N_E),
    return (W_in_mean, W_out_mean) in same units as g_EE.
    W_in is mean synaptic strength over synapses with both pre and post within the same assembly.
    """
    W_in_per_assembly, W_out_per_assembly = compute_W_in_W_out_per_assembly(g_EE, i_ee, j_ee, patterns)
    valid_in = [w for w in W_in_per_assembly if not np.isnan(w)]
    valid_out = [w for w in W_out_per_assembly if not np.isnan(w)]
    W_in_mean = np.nanmean(valid_in) if valid_in else np.nan
    W_out_mean = np.nanmean(valid_out) if valid_out else np.nan
    return W_in_mean, W_out_mean


def compute_W_in_W_out_per_assembly(g_EE, i_ee, j_ee, patterns):
    """
    Return (W_in_per_assembly, W_out_per_assembly).
    W_in_per_assembly: list of length n_patterns; element k is mean weight of synapses
    where both pre and post are in assembly k (nan if assembly has < 2 neurons).
    W_out_per_assembly: list of length n_patterns; element k is mean weight of synapses
    that have exactly one endpoint in assembly k (between assembly k and others).
    """
    n_patterns = patterns.shape[0]
    W_in_per_assembly = []
    W_out_per_assembly = []
    for k in range(n_patterns):
        in_k = np.where(patterns[k, :])[0]
        in_set = set(in_k)
        w_in_list = []
        w_out_list = []  # synapses with exactly one endpoint in this assembly
        for idx in range(len(g_EE)):
            pre, post = i_ee[idx], j_ee[idx]
            w = float(g_EE[idx]) if hasattr(g_EE[idx], '__float__') else g_EE[idx]
            pre_in = pre in in_set
            post_in = post in in_set
            if pre_in and post_in:
                w_in_list.append(w)
            elif pre_in or post_in:
                w_out_list.append(w)
        W_in_per_assembly.append(np.mean(w_in_list) if len(w_in_list) >= 2 else np.nan)
        W_out_per_assembly.append(np.mean(w_out_list) if w_out_list else np.nan)
    return W_in_per_assembly, W_out_per_assembly


def assembly_weights(results):
    """
    Compute average synaptic strength within assemblies (W_in) vs between (W_out).
    """
    g_EE = np.asarray(results['g_EE'])
    if hasattr(g_EE[0], 'item'):
        g_EE = np.array([float(x) for x in g_EE])
    w_in, w_out = compute_W_in_W_out(g_EE, results['i_ee'], results['j_ee'], results['patterns'])
    return {
        'W_in_mean': w_in,
        'W_out_mean': w_out,
        'W_in_per_pattern': [],
        'W_out_per_pattern': [],
    }


def fano_factor(spike_t, spike_i, neuron_indices, window_s, dt_s):
    """
    Fano factor = Var(count) / Mean(count) over windows for each neuron.
    """
    n_neurons = len(neuron_indices)
    n_windows = int(window_s / dt_s)
    if n_windows <= 0:
        return np.nan, np.nan
    counts = np.zeros((n_neurons, n_windows))
    t_max = spike_t.max() if len(spike_t) else 0
    for ni, n in enumerate(neuron_indices):
        mask = spike_i == n
        t = spike_t[mask]
        for w in range(n_windows):
            t_start = w * dt_s
            t_end = t_start + window_s
            counts[ni, w] = np.sum((t >= t_start) & (t < t_end))
    mean_c = np.nanmean(counts)
    var_c = np.nanvar(counts)
    if mean_c > 0:
        ff = var_c / mean_c
    else:
        ff = np.nan
    return ff, mean_c


def spike_count_correlation(spike_t, spike_i, neuron_pairs, window_s, dt_s):
    """
    Correlation of spike counts in windows for pairs of neurons.
    neuron_pairs: list of (i, j) indices.
    Returns mean correlation and std.
    """
    t_max = spike_t.max() if len(spike_t) else 0
    n_windows = max(1, int(t_max / dt_s))
    cors = []
    for (i, j) in neuron_pairs:
        ti = spike_t[spike_i == i]
        tj = spike_t[spike_i == j]
        c_i = np.zeros(n_windows)
        c_j = np.zeros(n_windows)
        for w in range(n_windows):
            t_start = w * dt_s
            t_end = t_start + window_s
            c_i[w] = np.sum((ti >= t_start) & (ti < t_end))
            c_j[w] = np.sum((tj >= t_start) & (tj < t_end))
        if np.std(c_i) > 0 and np.std(c_j) > 0:
            r = np.corrcoef(c_i, c_j)[0, 1]
            cors.append(r)
    return np.nanmean(cors) if cors else np.nan, np.nanstd(cors) if cors else np.nan


def run_analysis(results_path='results/assembly_simulation_results.pkl'):
    res = load_results(results_path)
    print('--- Assembly weights ---')
    w = assembly_weights(res)
    print(f"W_in (mean within assembly)  = {w['W_in_mean']:.4f} nS")
    print(f"W_out (mean between)         = {w['W_out_mean']:.4f} nS")
    print(f"W_in / W_out                 = {w['W_in_mean']/w['W_out_mean']:.4f}" if w['W_out_mean'] > 0 else "N/A")

    print('\n--- Fano factor (last 5 s of spontaneous, 100 ms windows) ---')
    t = res['sp_E_t']
    i = res['sp_E_i']
    # Use only last 5 s
    t_cut = t.max() - 5.0
    mask = t >= t_cut
    t = t[mask]
    i = i[mask]
    t -= t.min()
    n_neurons = res['params']['N_E']
    sample_neurons = np.linspace(0, n_neurons - 1, min(200, n_neurons), dtype=int)
    ff, mean_c = fano_factor(t, i, sample_neurons, window_s=0.1, dt_s=0.1)
    print(f"Fano factor = {ff:.4f} (mean count = {mean_c:.4f})")

    print('\n--- Spike count correlation (sample pairs, 100 ms windows) ---')
    pairs = [(sample_neurons[k], sample_neurons[k + 1]) for k in range(min(50, len(sample_neurons) - 1))]
    r_mean, r_std = spike_count_correlation(t, i, pairs, window_s=0.1, dt_s=0.1)
    print(f"Mean correlation = {r_mean:.4f} ± {r_std:.4f}")

    return {'weights': w, 'fano_factor': ff, 'correlation_mean': r_mean, 'correlation_std': r_std}


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/assembly_simulation_results.pkl'
    if not os.path.isfile(path):
        print(f'Run simulation first to generate {path}')
    else:
        run_analysis(path)