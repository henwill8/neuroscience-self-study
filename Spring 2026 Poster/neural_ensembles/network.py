"""
Build the Litwin-Kumar & Doiron style network: AdEx E, LIF I, conductance synapses,
triplet STDP (EE), row-sum normalization, iSTDP (EI), and Poisson drive.
"""
from brian2 import *
import numpy as np


def build_network(params, rng, plasticity_enabled=True, use_istdp=True):
    """
    Build neurons, synapses, Poisson input. Returns a dict with all objects and
    the pattern membership array (n_patterns x N_E) for stimulation.
    use_istdp: if False, J_EI is fixed at initial (for Figure 3 comparison).
    """
    p = params
    defaultclock.dt = p['dt']
    N_E, N_I = p['N_E'], p['N_I']
    delay_min = float(p.get('delay_min', 0*ms) / second)
    delay_max = float(p.get('delay_max', 1.5*ms) / second)

    # ----- Excitatory: AdEx with adaptive threshold (V_T + A_T, tau_T) -----
    adex_eqs = '''
        dv/dt = (g_L_E*(E_L_E - v) + g_L_E*Delta_T*exp((v - V_T_E)/Delta_T) - w + I_syn + I_ext) / C_E : volt
        dw/dt = (a_w*(v - E_L_E) - w) / tau_w : amp
        dV_thresh/dt = (V_T_E - V_thresh) / tau_T : volt
        dyE/dt = -yE / tau_y : 1
        dI_ext/dt = -I_ext / tau_ext : amp
        I_syn = g_E*(E_E - v) + g_I*(E_I - v) : amp
        dg_E/dt = (-g_E + h_E) / tau_d_E : siemens
        dh_E/dt = -h_E / tau_r_E : siemens
        dg_I/dt = (-g_I + h_I) / tau_d_I : siemens
        dh_I/dt = -h_I / tau_r_I : siemens
    '''
    # Reset: v to V_reset_E (Vre), w, yE; adaptive threshold reset to V_T_E + A_T; refractory t_abs
    V_reset_E = p.get('V_reset_E', -60*mV)
    refractory_E = p.get('refractory_E', 1*ms)
    reset_adex = 'v = V_reset_E; w += b_w; yE += 1; V_thresh = V_T_E + A_T'
    E_group = NeuronGroup(
        N_E,
        adex_eqs,
        threshold='v > V_thresh',
        reset=reset_adex,
        refractory=refractory_E,
        method='euler',
        namespace={
            'g_L_E': p['C_E'] / p['tau_E'],
            'E_L_E': p['E_L_E'],
            'V_T_E': p['V_T_E'],
            'V_reset_E': V_reset_E,
            'Delta_T': p['Delta_T'],
            'C_E': p['C_E'],
            'a_w': p['a_w'],
            'b_w': p['b_w'],
            'tau_w': p['tau_w'],
            'A_T': p.get('A_T', 10*mV),
            'tau_T': p.get('tau_T', 30*ms),
            'E_E': p['E_E'],
            'E_I': p['E_I'],
            'tau_r_E': p['tau_r_E'],
            'tau_d_E': p['tau_d_E'],
            'tau_r_I': p['tau_r_I'],
            'tau_d_I': p['tau_d_I'],
            'tau_y': p['tau_y'],
            'tau_ext': p.get('tau_ext', 2 * ms),
        },
    )
    E_group.v = p['E_L_E']
    E_group.w = 0 * amp
    E_group.V_thresh = p['V_T_E']
    E_group.yE = 0.0
    E_group.g_E = 0 * nS
    E_group.h_E = 0 * nS
    E_group.g_I = 0 * nS
    E_group.h_I = 0 * nS
    E_group.I_ext = 0 * amp

    # ----- Inhibitory: LIF with conductance and yI for iSTDP -----
    # LIF: do not declare shared params here so namespace values are used (avoids 0 from internal)
    lif_eqs = '''
        dv/dt = (g_L_I*(E_L_I - v) + I_syn + I_ext) / C_I : volt
        dyI/dt = -yI / tau_y : 1
        dI_ext/dt = -I_ext / tau_ext : amp
        I_syn = g_E*(E_E - v) + g_I*(E_I - v) : amp
        dg_E/dt = (-g_E + h_E) / tau_d_E : siemens
        dh_E/dt = -h_E / tau_r_E : siemens
        dg_I/dt = (-g_I + h_I) / tau_d_I : siemens
        dh_I/dt = -h_I / tau_r_I : siemens
    '''
    E_L_I = p['E_L_I']
    min_tau = 0.001 * ms
    I_group = NeuronGroup(
        N_I,
        lif_eqs,
        threshold='v >= V_T_I',
        reset='v = E_L_I; yI += 1',
        method='euler',
        namespace={
            'C_I': max(p['C_I'], 1 * pF),
            'g_L_I': p['C_I'] / p['tau_I'],
            'E_L_I': p['E_L_I'],
            'V_T_I': p['V_T_I'],
            'E_E': p['E_E'],
            'E_I': p['E_I'],
            'tau_r_E': max(p['tau_r_E'], min_tau),
            'tau_d_E': max(p['tau_d_E'], min_tau),
            'tau_r_I': max(p['tau_r_I'], min_tau),
            'tau_d_I': max(p['tau_d_I'], min_tau),
            'tau_y': max(p['tau_y'], min_tau),
            'tau_ext': max(p.get('tau_ext', 2 * ms), min_tau),
        },
    )
    I_group.v = p['E_L_I']
    I_group.yI = 0.0
    I_group.g_E = 0 * nS
    I_group.h_E = 0 * nS
    I_group.g_I = 0 * nS
    I_group.h_I = 0 * nS
    I_group.I_ext = 0 * amp

    # ----- Connectivity: random p_conn -----
    def random_conn(N_pre, N_post, p_conn, rng):
        n = int(round(N_pre * N_post * p_conn))
        idx = rng.choice(N_pre * N_post, size=min(n, N_pre * N_post), replace=False)
        i = np.unravel_index(idx, (N_pre, N_post))[0]
        j = np.unravel_index(idx, (N_pre, N_post))[1]
        return i, j

    # EE
    i_ee, j_ee = random_conn(N_E, N_E, p['p_conn'], rng)
    # No autapses
    no_aut = i_ee != j_ee
    i_ee, j_ee = i_ee[no_aut], j_ee[no_aut]
    n_ee = len(i_ee)

    # EI, IE, II
    i_ei, j_ei = random_conn(N_I, N_E, p['p_conn'], rng)
    i_ie, j_ie = random_conn(N_E, N_I, p['p_conn'], rng)
    i_ii, j_ii = random_conn(N_I, N_I, p['p_conn'], rng)
    no_aut_ii = i_ii != j_ii
    i_ii, j_ii = i_ii[no_aut_ii], j_ii[no_aut_ii]

    # ----- EE synapses: conductance + triplet STDP (Pfister-Gerstner style) -----
    # Traces:
    # r1/r2: pre traces (fast/slow), o1/o2: post traces (fast/slow)
    # on_pre: LTD
    # on_post: LTP
    tau_plus = p['triplet_tau_plus']
    tau_minus = p['triplet_tau_minus']
    tau_x = p['triplet_tau_x']
    tau_y_triplet = p['triplet_tau_y']
    A2_minus = p['triplet_A2_minus']
    A3_minus = p['triplet_A3_minus']
    A2_plus = p['triplet_A2_plus']
    A3_plus = p['triplet_A3_plus']
    J_EE_min = p['J_EE_min']
    J_EE_max = p['J_EE_max']

    ee_model = '''
        g_EE : siemens
        dr1/dt = -r1 / tau_plus : 1 (event-driven)
        dr2/dt = -r2 / tau_x : 1 (event-driven)
        do1/dt = -o1 / tau_minus : 1 (event-driven)
        do2/dt = -o2 / tau_y_triplet : 1 (event-driven)
        plasticity_on : 1 (constant)
    '''
    ee_on_pre = '''
        h_E_post += g_EE
        g_EE = clip(g_EE - plasticity_on * (A2_minus * o1 + A3_minus * o1 * r2), J_EE_min, J_EE_max)
        r1 += 1
        r2 += 1
    '''
    ee_on_post = '''
        g_EE = clip(g_EE + plasticity_on * (A2_plus * r1 + A3_plus * r1 * o2), J_EE_min, J_EE_max)
        o1 += 1
        o2 += 1
    '''

    Syn_EE = Synapses(
        E_group, E_group,
        model=ee_model,
        on_pre=ee_on_pre,
        on_post=ee_on_post,
        namespace={
            'tau_plus': tau_plus,
            'tau_minus': tau_minus,
            'tau_x': tau_x,
            'tau_y_triplet': tau_y_triplet,
            'A2_minus': A2_minus,
            'A3_minus': A3_minus,
            'A2_plus': A2_plus,
            'A3_plus': A3_plus,
            'J_EE_min': J_EE_min,
            'J_EE_max': J_EE_max,
        },
    )
    Syn_EE.connect(i=i_ee, j=j_ee)
    Syn_EE.delay = (delay_min + rng.random(n_ee) * (delay_max - delay_min)) * second
    Syn_EE.g_EE = p['g_EE_init']
    Syn_EE.r1 = 0
    Syn_EE.r2 = 0
    Syn_EE.o1 = 0
    Syn_EE.o2 = 0
    Syn_EE.plasticity_on = 1 if plasticity_enabled else 0

    if not plasticity_enabled:
        # Disable STDP by setting amplitudes to 0
        Syn_EE.namespace['A2_minus'] = 0 * nS
        Syn_EE.namespace['A3_minus'] = 0 * nS
        Syn_EE.namespace['A2_plus'] = 0 * nS
        Syn_EE.namespace['A3_plus'] = 0 * nS

    # Initial EE weights (for row-sum normalization in run_simulation).
    # Use constant initial value so we don't read from device (fails in cpp_standalone before run).
    g_EE_start = np.full(n_ee, float(p['g_EE_init']))

    # ----- EI: conductance + iSTDP -----
    r0 = float(p['r0'] / Hz)
    tau_y = float(p['tau_y'] / second)
    two_r0_tau_y = 2 * r0 * tau_y
    Z_EI = p['Z_EI']
    J_EI_min = p['J_EI_min']
    J_EI_max = p['J_EI_max']

    ei_model = '''
        g_EI : siemens
        plasticity_on : 1 (constant)
    '''
    ei_on_pre = '''
        h_I_post += g_EI
        g_EI = clip(g_EI + plasticity_on * Z_EI * (yE_post - two_r0_tau_y), J_EI_min, J_EI_max)
    '''
    ei_on_post = '''
        g_EI = clip(g_EI + plasticity_on * Z_EI * yI_pre, J_EI_min, J_EI_max)
    '''
    Syn_EI = Synapses(
        I_group, E_group,
        model=ei_model,
        on_pre=ei_on_pre,
        on_post=ei_on_post,
        namespace={'Z_EI': Z_EI, 'two_r0_tau_y': two_r0_tau_y, 'J_EI_min': J_EI_min, 'J_EI_max': J_EI_max},
    )
    Syn_EI.connect(i=i_ei, j=j_ei)
    n_ei = len(i_ei)
    Syn_EI.delay = (delay_min + rng.random(n_ei) * (delay_max - delay_min)) * second
    Syn_EI.g_EI = p['g_EI_init']
    Syn_EI.plasticity_on = 1 if (plasticity_enabled and use_istdp) else 0
    if not plasticity_enabled or not use_istdp:
        Syn_EI.namespace['Z_EI'] = 0 * nS

    # ----- IE: conductance only -----
    Syn_IE = Synapses(E_group, I_group, model='g_IE : siemens', on_pre='h_E_post += g_IE')
    Syn_IE.connect(i=i_ie, j=j_ie)
    n_ie = len(i_ie)
    Syn_IE.delay = (delay_min + rng.random(n_ie) * (delay_max - delay_min)) * second
    Syn_IE.g_IE = p['g_IE_init']

    # ----- II: conductance only -----
    Syn_II = Synapses(I_group, I_group, model='g_II : siemens', on_pre='h_I_post += g_II')
    Syn_II.connect(i=i_ii, j=j_ii)
    n_ii = len(i_ii)
    Syn_II.delay = (delay_min + rng.random(n_ii) * (delay_max - delay_min)) * second
    Syn_II.g_II = p['g_II_init']

    # ----- Stimulus patterns -----
    n_patterns = p['n_patterns']
    pattern_prob = p['pattern_prob']
    allow_pattern_overlap = p.get('allow_pattern_overlap', True)
    if allow_pattern_overlap:
        # Original behavior: independent Bernoulli membership for each pattern.
        patterns = (rng.random((n_patterns, N_E)) < pattern_prob)  # n_patterns x N_E, boolean
    else:
        # No-overlap mode: each neuron belongs to at most one pattern.
        patterns = np.zeros((n_patterns, N_E), dtype=bool)
        p_any = min(1.0, n_patterns * pattern_prob)
        assigned = rng.random(N_E) < p_any
        assigned_idx = np.where(assigned)[0]
        if assigned_idx.size > 0:
            target_pattern = rng.integers(0, n_patterns, size=assigned_idx.size)
            patterns[target_pattern, assigned_idx] = True

    # ----- Poisson drive: current-based with decay -----
    I_kick_E = p.get('I_kick_E', 0.05 * nA)
    I_kick_I = p.get('I_kick_I', 0.05 * nA)
    Poisson_E = PoissonGroup(N_E, rates=p['nu_E_baseline'])
    Poisson_I = PoissonGroup(N_I, rates=p['nu_I_baseline'])
    Syn_Poisson_E = Synapses(Poisson_E, E_group, on_pre='I_ext_post += I_kick_E', namespace={'I_kick_E': I_kick_E})
    Syn_Poisson_E.connect(i=np.arange(N_E), j=np.arange(N_E))
    Syn_Poisson_I = Synapses(Poisson_I, I_group, on_pre='I_ext_post += I_kick_I', namespace={'I_kick_I': I_kick_I})
    Syn_Poisson_I.connect(i=np.arange(N_I), j=np.arange(N_I))

    return {
        'E_group': E_group,
        'I_group': I_group,
        'Syn_EE': Syn_EE,
        'Syn_EI': Syn_EI,
        'Syn_IE': Syn_IE,
        'Syn_II': Syn_II,
        'Poisson_E': Poisson_E,
        'Poisson_I': Poisson_I,
        'Syn_Poisson_E': Syn_Poisson_E,
        'Syn_Poisson_I': Syn_Poisson_I,
        'patterns': patterns,
        'i_ee': i_ee,
        'j_ee': j_ee,
        'i_ei': i_ei,
        'j_ei': j_ei,
        'g_EE_start': g_EE_start,
    }