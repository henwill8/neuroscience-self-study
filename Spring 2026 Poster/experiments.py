"""
Experiments: one function per experiment. Each creates and runs a network with optional param overrides.
Run with:  python experiments.py [--list] [experiment_name ...]
  e.g.    python experiments.py
          python experiments.py default
          python experiments.py default template
          python experiments.py --list
"""
import argparse
from brian2 import seed
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
    if show_plots:
        plot_all_figures(results, show=True)
    return params, results


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@_register("default", "Default network: full default params, 50 trials, all figures.")
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
