"""
Experiments: one function per experiment. Each creates and runs a network with optional param overrides.
Run with:  python experiments.py [--list] [experiment_name ...]
  e.g.    python experiments.py
          python experiments.py default
          python experiments.py default template
          python experiments.py --list
"""
import argparse
from brian2 import seed, second
import numpy as np

from config import get_default_params, derive_trial_params
from network import Network
from plotting import SimpleResults, plot_all_figures


# ---------------------------------------------------------------------------
# Experiment registry: name -> (short_description, run_function)
# ---------------------------------------------------------------------------
EXPERIMENTS = {}


def _register(name, description):
    """Decorator: register an experiment function under the given name and description."""
    def decorator(fn):
        EXPERIMENTS[name] = (description, fn)
        return fn
    return decorator


def _run_network(params, rng, show_plots=True):
    """Build and run network with given params and rng; optionally show all figures. Returns (params, results)."""
    net = Network(params, rng)
    params, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh = net.run()
    results = SimpleResults(
        spikeMonExc, spikeMonInh, stateMonExc, stateMonInh, params
    )
    if params.get('measure_ns_peak_firing', False):
        stats = results.get_ns_peak_firing_stats()
        if stats is not None:
            results.p['ns_peak_firing_stats'] = stats
            print(f"NS (non-stimulated) excitatory: mean peak time = {stats['mean_peak_time_s']:.4f} s, "
                  f"variance = {stats['variance_peak_time_s']:.6f} s² (n_ns = {stats['n_ns_neurons']})")
    if show_plots:
        plot_all_figures(results, show=True)
    return params, results


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@_register("default", "Default network: full default params.")
def run_default(show_plots=True):
    """Run the standard network with no param overrides."""
    seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    params = get_default_params()
    return _run_network(params, rng, show_plots=show_plots)


@_register("template", "Template: fewer trials and units for quick testing. Copy this to add new experiments.")
def run_template(show_plots=True):
    """
    Template experiment with overridden params. Copy this function and the decorator
    to add a new experiment; override any keys in get_default_params(), then call
    derive_trial_params(params) if you changed trial-related params.
    """
    seed(43)
    np.random.seed(43)
    rng = np.random.default_rng(43)
    params = get_default_params()
    # Override for a quick run (fewer trials, smaller network)
    params['nTrials'] = 5
    params['nUnits'] = 500
    params['n_record_voltage'] = 20
    derive_trial_params(params)  # required when changing nTrials, interTrialInterval, etc.
    return _run_network(params, rng, show_plots=show_plots)


@_register("continue", "Load weights from a checkpoint; all other params from defaults (or overrides).")
def run_continue(show_plots=True, checkpoint_path="results/network_checkpoint.pkl", **param_overrides):
    """
    Load only weights from checkpoint_path. Params come from get_default_params(); pass any overrides as kwargs.
    Network size (nExc, nInh) must match the checkpoint.

    Example (from code):
      run_continue(show_plots=False, checkpoint_path="results/ckpt.pkl", nTrials=20)
    """
    seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    params = get_default_params()
    params['load_checkpoint_path'] = checkpoint_path
    for key, value in param_overrides.items():
        params[key] = value
    derive_trial_params(params)
    return _run_network(params, rng, show_plots=show_plots)


@_register("continue_custom", "Example: load weights from checkpoint and set params in the function body (copy to add your own).")
def run_continue_custom(show_plots=True):
    """
    Load weights from a checkpoint; set params in this function. Copy and edit for your own experiment.
    """
    seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    params = get_default_params()
    params['load_checkpoint_path'] = "results/network_checkpoint.pkl"
    params['nTrials'] = 10
    params['checkpoint_path'] = 'results/continued_experiment.pkl'
    derive_trial_params(params)
    return _run_network(params, rng, show_plots=show_plots)


@_register("ns_perturb_trial0", "Temporary test: perturb NS neurons on first trial 0.2 s after start.")
def run_ns_perturb_trial0(show_plots=True):
    """
    Perturb all NS (non-stimulated) neurons with a single pulse 0.2 s after trial 0 start.
    Uses same current amplitude as CS/US by default. Compare NS trajectories with/without.
    """
    seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    params = get_default_params()
    params['ns_perturbation_trial'] = 0
    params['ns_perturbation_t_s'] = 0.2  # seconds after trial 0 start
    params['ns_perturbation_amplitude'] = 0.2  # nA; optional, defaults to spikeInputAmplitude
    derive_trial_params(params)
    return _run_network(params, rng, show_plots=show_plots)


@_register("upstate_kick", "Transient kick to random small % of E neurons to probe up-state initiation and structured spontaneity.")
def run_upstate_kick(show_plots=True):
    """
    Spontaneous activity only: no CS/US trials. Deliver a brief external kick to a random 2%
    of excitatory neurons at a fixed time (after a short baseline). Goal: see whether the kick
    can initiate an up state and whether subsequent activity is structured or unstructured.
    """
    seed(43)
    np.random.seed(43)
    rng = np.random.default_rng(43)
    params = get_default_params()
    # No trials — spontaneous activity only, then one kick
    params['nTrials'] = 0
    params['include_CS_only_trial'] = False
    derive_trial_params(params)
    # Explicit duration (derive_trial_params would give invalid duration when nTrials=0)
    params['duration'] = 5 * second  # ~2 s baseline, kick at 2 s, ~3 s to observe response
    # Kick after ~2 s baseline
    params['upstate_kick_t_s'] = 2.0
    params['upstate_kick_fraction'] = 0.02   # 2% of E neurons
    params['upstate_kick_amplitude'] = 0.5   # nA (strong enough to depolarize)
    params['upstate_kick_n_pulses'] = 5      # short train at 50 Hz
    params['upstate_kick_Hz'] = 50.0
    params['save_checkpoint'] = False
    params['load_checkpoint_path'] = 'results/istdp_network_checkpoint.pkl'
    params, results = _run_network(params, rng, show_plots=show_plots)
    if params.get('upstate_kick_n'):
        print(f"Up-state kick: {params['upstate_kick_n']} neurons at t={params.get('upstate_kick_t_s', '?')} s")
    return params, results


# ---------------------------------------------------------------------------
# CLI: choose which experiments to run
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run one or more network experiments. Each experiment builds and runs a network (with optional param overrides).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments.py              Run the default experiment
  python experiments.py default      Same as above
  python experiments.py template    Run the template (quick test) experiment
  python experiments.py default template   Run both, in order
  python experiments.py --list      Show available experiments
        """,
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        default=["default"],
        metavar="NAME",
        help="Experiment name(s) to run. Default: default. Use --list to see names.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiments and exit.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run without showing figures (faster for batch runs).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:\n")
        for name, (desc, _) in sorted(EXPERIMENTS.items()):
            print(f"  {name:<12}  {desc}")
        print()
        return

    show_plots = not args.no_plot
    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        parser.error(f"Unknown experiment(s): {', '.join(unknown)}. Use --list to see options.")

    for i, name in enumerate(args.experiments):
        if i > 0:
            print()
        desc, run_fn = EXPERIMENTS[name]
        print(f"[{name}] {desc}")
        run_fn(show_plots=show_plots)
        print(f"[{name}] done.")


if __name__ == "__main__":
    main()
