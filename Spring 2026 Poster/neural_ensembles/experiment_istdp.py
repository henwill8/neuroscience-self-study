"""
iSTDP experiment: run simulation with inhibitory STDP enabled, then show raster and W_in/W_out.
Usage:
  python experiment_istdp.py           # full protocol (slow)
  python experiment_istdp.py --quick   # shorter run for testing
  python experiment_istdp.py --small   # smaller network
"""
import os
import sys

# Ensure we can import from this package when run from any cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from brian2 import second
from run_simulation import run_full_simulation
from figures_lk2014 import plot_fig2b, plot_full_raster_single, FIG_DIR, RESULTS_DIR


def main():
    args = set(sys.argv[1:])
    quick = '--quick' in args
    small = '--small' in args

    # Run with iSTDP only
    n_block_repeats = 2 if quick else 20
    spontaneous_duration_s = 5.0 if quick else 120.0
    record_interval_s = 0.5 if quick else 1.0
    params_overrides = {}
    if quick:
        params_overrides['t_warmup_no_plasticity'] = 1 * second
    if small:
        params_overrides.update({'N_E': 2000, 'N_I': 500, 'n_patterns': 3})

    print('Running iSTDP simulation...')
    run_full_simulation(
        n_block_repeats=n_block_repeats,
        spontaneous_duration_s=spontaneous_duration_s,
        record_interval_s=record_interval_s,
        use_istdp=True,
        params_overrides=params_overrides if params_overrides else None,
    )

    results_path = os.path.join(RESULTS_DIR, 'assembly_with_istdp.pkl')
    if not os.path.isfile(results_path):
        print('Error: results not saved to %s' % results_path)
        return

    print('Plotting W_in / W_out...')
    plot_fig2b(results_path=results_path, save=True)

    print('Plotting full raster...')
    plot_full_raster_single(results_path=results_path, save=True, title='iSTDP experiment: full raster')

    print('Done. Figures saved to %s/' % FIG_DIR)


if __name__ == '__main__':
    main()
