# Neuronal Assemblies Simulation (Litwin-Kumar & Doiron style)

Brian2 simulation for the formation of neuronal assemblies via voltage-based excitatory STDP (Clopath et al.), inhibitory STDP (iSTDP), and synaptic normalization.

## Network

- **4,000** excitatory (AdEx) and **1,000** inhibitory (LIF) neurons
- Random connectivity **p = 0.2**
- Conductance-based synapses with double-exponential kinetics (τ_r, τ_d for E and I)
- **EE**: Clopath voltage-based STDP (u, ū, x), bounds [J_EE_min, J_EE_max]
- **EE row-sum normalization** every 20 ms
- **EI**: iSTDP targeting 3 Hz postsynaptic rate

## Protocol

1. **Warmup**: 10 s with plasticity off (transients settle)
2. **Training**: 20 patterns × (1 s stimulus + 3 s gap), repeated for 20 blocks. Each pattern stimulates a random 5% of E neurons; external Poisson +8 kHz during stimulus
3. **Spontaneous**: Baseline Poisson only (4.5 kHz E, 2.25 kHz I)

## Usage

```bash
cd neural_ensembles
python run_simulation.py         # Full run (4000 E / 1000 I) — needs C++ compiler for speed
python run_simulation.py --small # Smaller/faster run (500 E / 125 I) for testing
python run_simulation.py --quick # Minimal run (~1 min)
python run_simulation.py --no-istdp   # For Fig 3 comparison (no iSTDP)
python analysis.py              # W_in/W_out, Fano factor, correlations (after run)
```

**Speed:** The full network (~3.2M EE synapses) is very slow without compiled code. If the script prints `codegen target: numpy`, install a C++ compiler and Cython (see [Brian2 installation](https://brian2.readthedocs.io/en/stable/introduction/install.html)); otherwise use `--small` to run a reduced network in a few minutes.

## Outputs

- `results/assembly_simulation_results.pkl`: weights (g_EE, g_EE_start), patterns, spike times (sp_E_t, sp_E_i, sp_I_t, sp_I_i)
- Analysis: W_in vs W_out (within vs between assemblies), Fano factor, spike count correlations

## Parameters

Edit `config.py` for network size, plasticity bounds, stimulation protocol, and baseline/stimulus rates.