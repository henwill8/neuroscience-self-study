"""
Parameters aligned with Litwin-Kumar & Doiron (2014) and the authors' reference code
(`litwin-kumar_doiron_formation_2014/sim.jl`). Table values use nS where noted; initial
EE weight jee0 follows sim.jl (2.86), not the 2.76 line in some paper tables.
"""
from brian2 import *

def get_params():
    return {
        'dt': 0.1 * ms,
        'seed': 42,

        # -------------------------------------------------------------------------
        # Population sizes [Table 1 / sim.jl]
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
        'A_T': 10 * mV,
        'tau_T': 30 * ms,
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
        # Connectivity and synapses [Table 2 / sim.jl weightpars]
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
        # sim.jl: jee0 = 2.86 (initial EE); jie, jii, jei0 as below
        'g_EE_init': 2.86 * nS,
        'g_IE_init': 1.27 * nS,
        'g_II_init': 16.2 * nS,
        'g_EI_init': 48.7 * nS,

        # -------------------------------------------------------------------------
        # Excitatory voltage-based STDP [sim.jl: altd, altp, thetaltd, thetaltp, tauu, tauv, taux]
        # LTD on pre spike; LTP on post spike (event-driven in network.py; faster than Python scans).
        # -------------------------------------------------------------------------
        'J_EE_min': 1.78 * nS,
        'J_EE_max': 21.4 * nS,
        'tau_u': 10 * ms,
        'tau_u_bar': 7 * ms,
        'tau_x': 15 * ms,
        'theta_LTD': -70 * mV,
        'theta_LTP': -49 * mV,
        'A_LTD': 0.0008 * nS,
        'A_LTP': 0.0014 * nS,

        # -------------------------------------------------------------------------
        # Homeostatic normalization: sim.jl dtnormalize = 20 (ms) each step dt=0.1 ms
        # -------------------------------------------------------------------------
        'norm_period': 20 * ms,

        # -------------------------------------------------------------------------
        # Inhibitory STDP [sim.jl: eta=1, r0=0.003 kHz, tauy=20 ms]
        # -------------------------------------------------------------------------
        'r0': 3 * Hz,
        'tau_y': 20 * ms,
        'Z_EI': 1.0 * nS,
        'J_EI_min': 48.7 * nS,
        'J_EI_max': 243 * nS,

        # -------------------------------------------------------------------------
        # External drive [sim.jl rex, rix in kHz] — kick amplitudes are a Brian2 stand-in
        # for jex/jix pulse increments in the reference integrator.
        # -------------------------------------------------------------------------
        'nu_E_baseline': 4.5 * kHz,
        'nu_I_baseline': 2.25 * kHz,
        'nu_stim_extra': 8 * kHz,
        'tau_ext': 2 * ms,
        'I_kick_E': 0.05 * nA,
        'I_kick_I': 0.05 * nA,

        # -------------------------------------------------------------------------
        # Protocol (paper Fig. 2 style)
        # -------------------------------------------------------------------------
        'n_patterns': 20,
        'pattern_prob': 0.05,
        # sim.jl simnew: independent membership draw per assembly (overlap allowed).
        'allow_pattern_overlap': True,
        'pattern_duration': 1 * second,
        'inter_pattern_gap': 3 * second,
        'n_block_repeats': 20,

        # Paper Methods: 10 s without STDP. sim.jl uses stdpdelay=1000 (1 s if t is in ms).
        't_warmup_no_plasticity': 10 * second,
    }
