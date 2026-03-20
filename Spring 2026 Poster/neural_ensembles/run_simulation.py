"""
Run the full L&D 2014 protocol: warmup -> training (with W_in/W_out recording) -> spontaneous.
Supports use_istdp=False for Figure 3 comparison.

Speed: With the default 4000 E / 1000 I network (~3.2M EE synapses), the simulation requires
compiled code to run in reasonable time. If you see "codegen target: numpy", install a C++
compiler and Cython (see Brian2 docs); otherwise use --small for a smaller, faster run.
"""
from brian2 import *
import numpy as np
import pickle
import os

from config import get_params
from network import build_network
from analysis import compute_W_in_W_out, compute_W_in_W_out_per_assembly


def _setup_codegen(use_cython=True):
    """Prefer Cython for speed; fall back to numpy if Brian2's Cython isn't available."""
    # Brian2's extension_manager imports Cython at brian2 load time; if that failed, it sets
    # Cython = None and will raise "Cython is not available" when compiling. So we must check
    # the extension_manager's Cython, not just whether we can import Cython here.
    if use_cython:
        try:
            from brian2.codegen.runtime.cython_rt import extension_manager as _cython_em
            if getattr(_cython_em, 'Cython', None) is None:
                use_cython = False
            else:
                prefs.codegen.target = 'cython'
        except Exception:
            use_cython = False
    if not use_cython:
        prefs.codegen.target = 'numpy'
    target = prefs.codegen.target
    print('Brian2 codegen target: %s' % target)
    if target == 'numpy':
        print('  WARNING: Running with numpy (Cython not available to Brian2). Very slow for full-size network.')
        print('  Use --small or --quick for reduced runs. To fix Cython: pip install cython and ensure Cython.Build imports.')


def run_full_simulation(
    n_block_repeats=20,
    spontaneous_duration_s=120.0,
    save_dir=None,
    rng_seed=None,
    params_overrides=None,
    use_istdp=True,
    record_interval_s=1.0,
):
    """
    use_istdp: if False, J_EI fixed at initial (Fig 3 comparison).
    record_interval_s: sample W_in/W_out every this many seconds.
    save_dir: if None, use neural_ensembles/results (relative to this script) so the same
              folder is used regardless of current working directory.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    p = get_params()
    if params_overrides:
        p.update(params_overrides)
    if rng_seed is not None:
        p['seed'] = rng_seed
    seed(p['seed'])
    np.random.seed(p['seed'])
    rng = np.random.default_rng(p['seed'])

    p['n_block_repeats'] = n_block_repeats
    N_E = p['N_E']
    patterns = None

    _setup_codegen(use_cython=True)

    # This simulation uses Python callbacks (row-sum norm) and changing Poisson rates between
    # run() calls, which cpp_standalone does not support. Use runtime (cython) only.
    try:
        from brian2 import get_device
        if get_device().name == 'cpp_standalone':
            set_device('runtime')
            print('Note: cpp_standalone not supported (Python norm + multi-phase run); using runtime.')
    except Exception:
        pass

    net = build_network(p, rng, plasticity_enabled=True, use_istdp=use_istdp)
    patterns = net['patterns']
    Syn_EE = net['Syn_EE']
    Syn_EI = net['Syn_EI']
    Poisson_E = net['Poisson_E']
    E_group = net['E_group']
    I_group = net['I_group']
    j_ee = net['j_ee']
    i_ee = net['i_ee']
    g_EE_start = net['g_EE_start']

    Syn_EE.plasticity_on = 0
    Syn_EI.plasticity_on = 0

    norm_period = p['norm_period']
    nExc = N_E
    # Only normalize when plasticity is on
    normalization_active = [False]

    def _norm_op():
        if not normalization_active[0]:
            return
        syn = Syn_EE
        post_inds = np.asarray(syn.j)
        j_cur = np.asarray(syn.g_EE)
        j_start = g_EE_start
        # L&D: J_EE_ij(t) <- J_EE_ij(t) - (sum_j J_EE_ij(t) - sum_j J_EE_ij(0)) / N_i
        # N_i = number of incoming connections to post i
        sum_cur = np.bincount(post_inds, weights=j_cur, minlength=nExc)
        sum_start = np.bincount(post_inds, weights=j_start, minlength=nExc)
        N_i = np.bincount(post_inds, minlength=nExc)
        excess = sum_cur - sum_start
        delta_i = np.zeros(nExc)
        np.divide(excess, N_i, where=N_i > 0, out=delta_i)
        new_j = j_cur - delta_i[post_inds]
        # Clamp to bounds
        J_min = float(p['J_EE_min'])
        J_max = float(p['J_EE_max'])
        new_j = np.clip(new_j, J_min, J_max)
        syn.g_EE[:] = new_j * syn.g_EE.unit

    norm_op = network_operation(dt=norm_period, when='start')(_norm_op)

    sp_E = SpikeMonitor(E_group)
    sp_I = SpikeMonitor(I_group)

    objects = [
        net['E_group'], net['I_group'],
        net['Syn_EE'], net['Syn_EI'], net['Syn_IE'], net['Syn_II'],
        net['Poisson_E'], net['Poisson_I'],
        net['Syn_Poisson_E'], net['Syn_Poisson_I'],
        sp_E, sp_I, norm_op,
    ]
    b2_net = Network(*objects)
    report_period = 5 * second

    # ----- Phase 1: Warmup -----
    print('Phase 1: Warmup (plasticity off)')
    b2_net.run(p['t_warmup_no_plasticity'], report='text', report_period=report_period)

    Syn_EE.plasticity_on = 1
    Syn_EI.plasticity_on = 1
    normalization_active[0] = True  # enable row-sum norm now that plasticity is on
    print('Plasticity enabled (iSTDP=%s).' % use_istdp)

    # ----- Phase 2: Training (run in chunks, record W_in/W_out) -----
    # Chunks are pattern_dur (stimulus on) and gap_dur (baseline): we must switch Poisson
    # rates between each chunk, so Phase 2 is many short runs rather than one long run.
    pattern_dur = p['pattern_duration']
    gap_dur = p['inter_pattern_gap']
    n_patterns = p['n_patterns']
    nu_baseline = p['nu_E_baseline']
    nu_stim_extra = p['nu_stim_extra']
    rates_baseline = np.ones(N_E) * float(nu_baseline / Hz)
    record_dt = record_interval_s * second

    phase2_total_s = p['n_block_repeats'] * n_patterns * (float(pattern_dur / second) + float(gap_dur / second))
    phase2_start_s = float(defaultclock.t / second)
    phase_report_interval_s = 5.0
    phase2_next_report_s = phase_report_interval_s  # first update at 10 s

    W_in_t = []
    W_in_per_assembly_vals = []
    W_out_per_assembly_vals = []
    t_elapsed = defaultclock.t

    print('Phase 2: Training %.0f s total (recording W_in/W_out every %.1f s)' % (phase2_total_s, record_interval_s))
    for block in range(p['n_block_repeats']):
        for k in range(n_patterns):
            rates_stim = rates_baseline + patterns[k, :] * float(nu_stim_extra / Hz)
            Poisson_E.rates = rates_stim * Hz
            b2_net.run(pattern_dur)
            t_elapsed = float(defaultclock.t / second)
            phase2_elapsed_s = t_elapsed - phase2_start_s
            while phase2_elapsed_s >= phase2_next_report_s and phase2_next_report_s <= phase2_total_s:
                pct = 100.0 * phase2_next_report_s / phase2_total_s if phase2_total_s > 0 else 0
                print('  Phase 2: %.1f / %.1f s (%.1f%%)' % (phase2_next_report_s, phase2_total_s, pct))
                phase2_next_report_s += phase_report_interval_s
            if record_interval_s and (len(W_in_t) == 0 or t_elapsed >= W_in_t[-1] + record_interval_s):
                g_cur = np.asarray(Syn_EE.g_EE[:])
                wi_per_asm, wo_per_asm = compute_W_in_W_out_per_assembly(g_cur, i_ee, j_ee, patterns)
                W_in_t.append(t_elapsed)
                W_in_per_assembly_vals.append(wi_per_asm)
                W_out_per_assembly_vals.append(wo_per_asm)
            Poisson_E.rates = nu_baseline
            b2_net.run(gap_dur)
            t_elapsed = float(defaultclock.t / second)
            phase2_elapsed_s = t_elapsed - phase2_start_s
            while phase2_elapsed_s >= phase2_next_report_s and phase2_next_report_s <= phase2_total_s:
                pct = 100.0 * phase2_next_report_s / phase2_total_s if phase2_total_s > 0 else 0
                print('  Phase 2: %.1f / %.1f s (%.1f%%)' % (phase2_next_report_s, phase2_total_s, pct))
                phase2_next_report_s += phase_report_interval_s
            if record_interval_s and (len(W_in_t) == 0 or t_elapsed >= W_in_t[-1] + record_interval_s):
                g_cur = np.asarray(Syn_EE.g_EE[:])
                wi_per_asm, wo_per_asm = compute_W_in_W_out_per_assembly(g_cur, i_ee, j_ee, patterns)
                W_in_t.append(t_elapsed)
                W_in_per_assembly_vals.append(wi_per_asm)
                W_out_per_assembly_vals.append(wo_per_asm)

    t_training_end = float(defaultclock.t / second)
    print('  Phase 2: %.1f / %.1f s (100%%) done' % (phase2_total_s, phase2_total_s))

    # ----- Phase 3: Spontaneous -----
    phase3_total_s = spontaneous_duration_s
    phase3_next_report_s = phase_report_interval_s  # first update at 10 s
    print('Phase 3: Spontaneous %.0f s total' % phase3_total_s)
    Poisson_E.rates = nu_baseline
    t_end = t_training_end + spontaneous_duration_s
    while float(defaultclock.t / second) < t_end:
        run_dur = min(record_dt, (t_end * second - defaultclock.t))
        b2_net.run(run_dur)
        t_elapsed = float(defaultclock.t / second)
        phase3_elapsed_s = t_elapsed - t_training_end
        while phase3_elapsed_s >= phase3_next_report_s and phase3_next_report_s <= phase3_total_s:
            pct = 100.0 * phase3_next_report_s / phase3_total_s if phase3_total_s > 0 else 0
            print('  Phase 3: %.1f / %.1f s (%.1f%%)' % (phase3_next_report_s, phase3_total_s, pct))
            phase3_next_report_s += phase_report_interval_s
        if record_interval_s and (len(W_in_t) == 0 or t_elapsed >= W_in_t[-1] + record_interval_s):
            g_cur = np.asarray(Syn_EE.g_EE[:])
            wi_per_asm, wo_per_asm = compute_W_in_W_out_per_assembly(g_cur, i_ee, j_ee, patterns)
            W_in_t.append(t_elapsed)
            W_in_per_assembly_vals.append(wi_per_asm)
            W_out_per_assembly_vals.append(wo_per_asm)

    print('  Phase 3: %.1f / %.1f s (100%%) done' % (phase3_total_s, phase3_total_s))

    g_EE = np.asarray(Syn_EE.g_EE[:])
    g_EI = np.asarray(Syn_EI.g_EI[:]) if hasattr(Syn_EI.g_EI, '__len__') else None
    i_ei = net['i_ei']
    j_ei = net['j_ei']

    os.makedirs(save_dir, exist_ok=True)
    suffix = 'no_istdp' if not use_istdp else 'with_istdp'
    t_warmup_s = float(p['t_warmup_no_plasticity'] / second)
    results = {
        'params': p,
        'i_ee': i_ee, 'j_ee': j_ee,
        'i_ei': i_ei, 'j_ei': j_ei,
        'g_EE': g_EE, 'g_EE_start': g_EE_start,
        'g_EI': g_EI,
        'patterns': patterns,
        'sp_E_t': np.asarray(sp_E.t / second),
        'sp_E_i': np.asarray(sp_E.i),
        'sp_I_t': np.asarray(sp_I.t / second),
        'sp_I_i': np.asarray(sp_I.i),
        'W_in_t': np.array(W_in_t),
        'W_in_per_assembly_vals': W_in_per_assembly_vals,
        'W_out_per_assembly_vals': W_out_per_assembly_vals,
        't_training_end': t_training_end,
        't_warmup_s': t_warmup_s,
        'phase2_total_s': phase2_total_s,
        'spontaneous_duration_s': spontaneous_duration_s,
        'use_istdp': use_istdp,
    }
    path = os.path.join(save_dir, 'assembly_%s.pkl' % suffix)
    with open(path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to %s' % path)
    return results


if __name__ == '__main__':
    import sys
    args = set(sys.argv[1:])

    # --cpp: request C++ standalone; script will fall back to runtime (this sim needs Python norm + multi-phase run)
    if '--cpp' in args:
        args.remove('--cpp')
        set_device('cpp_standalone', build_on_run=False)

    # Compounding: any combination of --quick, --small, --no-istdp, plus optional --cpp
    use_istdp = '--no-istdp' not in args

    # Default run (full protocol)
    n_block_repeats = 20
    spontaneous_duration_s = 120.0
    record_interval_s = 1.0
    params_overrides = {}

    if '--small' in args:
        params_overrides.update({
            'N_E': 2000,
            'N_I': 500,
            'n_patterns': 3,
        })

    if '--quick' in args:
        n_block_repeats = 2
        spontaneous_duration_s = 5.0
        record_interval_s = 0.5
        params_overrides.update({
            't_warmup_no_plasticity': 1 * second,
            'pattern_duration': 1 * second,
            'inter_pattern_gap': 3 * second,
            'dt': 0.1 * ms,
        })

    run_full_simulation(
        n_block_repeats=n_block_repeats,
        spontaneous_duration_s=spontaneous_duration_s,
        record_interval_s=record_interval_s,
        use_istdp=use_istdp,
        params_overrides=params_overrides if params_overrides else None,
    )
