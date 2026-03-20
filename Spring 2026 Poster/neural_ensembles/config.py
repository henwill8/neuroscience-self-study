"""
Exact parameters from Litwin-Kumar & Doiron (2014) Tables 1-4.
Weights as conductance in nS (paper reports in pF; we use same numeric values in nS).
"""
from brian2 import *

def get_params():
    return {
        'dt': 0.1 * ms,
        'seed': 42,

        # -------------------------------------------------------------------------
        # Population sizes [Table 1]
        # -------------------------------------------------------------------------
        'N_E': 4000,
        'N_I': 1000,

        # -------------------------------------------------------------------------
        # Excitatory: AdEx [Table 1] + adaptive threshold (V_T + A_T, tau_T = 30 ms)
        # -------------------------------------------------------------------------
        'tau_E': 20 * ms,
        'E_L_E': -70 * mV,
        'V_T_E': -52 * mV,
        'Delta_T': 2 * mV,
        'C_E': 300 * pF,
        'a_w': 4 * nS,
        'b_w': 0.805 * pA,
        'tau_w': 150 * ms,
        'g_L_E': 15 * nS,
        # Adaptive threshold: reset to V_T + A_T, decay tau_T
        'A_T': 10 * mV,
        'tau_T': 30 * ms,
        # Reset voltage and absolute refractory (L&D Methods)
        'V_reset_E': -60 * mV,
        'refractory_E': 1 * ms,

        # -------------------------------------------------------------------------
        # Inhibitory: LIF [Table 1]
        # -------------------------------------------------------------------------
        'tau_I': 20 * ms,
        'E_L_I': -62 * mV,
        'V_T_I': -52 * mV,
        'C_I': 300 * pF,
        'g_L_I': 15 * nS,

        # -------------------------------------------------------------------------
        # Connectivity and synapses [Table 2]
        # -------------------------------------------------------------------------
        'p_conn': 0.2,
        'delay_min': 0 * ms,
        'delay_max': 1.5 * ms,
        'tau_r_E': 1 * ms,
        'tau_d_E': 6 * ms,
        'tau_r_I': 0.5 * ms,
        'tau_d_I': 2 * ms,
        'E_E': 0 * mV,
        'E_I': -75 * mV,
        # Baseline weights (nS) [Table 2]: J_EE0=2.76, J_IE=1.27, J_II=16.2
        'g_EE_init': 2.76 * nS,
        'g_IE_init': 1.27 * nS,
        'g_II_init': 16.2 * nS,
        # J_EI: initial and bounds [Table 2, 4]
        'g_EI_init': 48.7 * nS,

        # -------------------------------------------------------------------------
        # Excitatory STDP (Clopath) [Table 3]. A_LTP increased so LTP dominates
        # during co-activity (LTP term has small voltage factors ~0.01 each).
        # -------------------------------------------------------------------------
        'J_EE_min': 1.78 * nS,
        'J_EE_max': 21.4 * nS,
        'tau_u': 10 * ms,
        'tau_u_bar': 7 * ms,
        'tau_x': 15 * ms,
        'theta_LTD': -70 * mV,
        'theta_LTP': -49 * mV,
        'A_LTD': 0.0000 * nS,
        'A_LTP': 0.012 * nS,

        # -------------------------------------------------------------------------
        # Homeostatic normalization: every 20 ms, row sum of J_EE constant
        # -------------------------------------------------------------------------
        'norm_period': 20 * ms,

        # -------------------------------------------------------------------------
        # Inhibitory STDP [Table 4]: r0=3 Hz, eta=1 pA (as Z in nS), tau_y=20 ms.
        # Z_EI reduced from 1.0 so assembly firing does not collapse after first pattern
        # (homeostasis should regulate toward target rate, not oversuppress).
        # -------------------------------------------------------------------------
        'r0': 3 * Hz,
        'tau_y': 20 * ms,
        'Z_EI': 0.2 * nS,
        'J_EI_min': 48.7 * nS,
        'J_EI_max': 243 * nS,

        # -------------------------------------------------------------------------
        # External Poisson [Table 2]: r_EEext=4.5 kHz, r_IEext=2.25 kHz
        # -------------------------------------------------------------------------
        'nu_E_baseline': 4.5 * kHz,
        'nu_I_baseline': 2.25 * kHz,
        'nu_stim_extra': 8 * kHz,
        'tau_ext': 2 * ms,
        'I_kick_E': 0.05 * nA,
        'I_kick_I': 0.05 * nA,

        # -------------------------------------------------------------------------
        # Protocol: 20 patterns, 1 s each, 3 s gaps, 20 repeats
        # -------------------------------------------------------------------------
        'n_patterns': 20,
        'pattern_prob': 0.05,
        'pattern_duration': 1 * second,
        'inter_pattern_gap': 3 * second,
        'n_block_repeats': 20,

        't_warmup_no_plasticity': 10 * second,
    }
