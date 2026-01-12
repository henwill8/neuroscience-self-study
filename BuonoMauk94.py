from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def population_vector(spike_monitor, t_start, t_end, N):
    """
    Returns a binary population vector of length N
    """
    vec = np.zeros(N, dtype=int)
    mask = (spike_monitor.t >= t_start) & (spike_monitor.t < t_end)
    active_neurons = np.unique(spike_monitor.i[mask])
    vec[active_neurons] = 1
    return vec

def percent_active(spike_monitor, t_bins, N):
    """
    Returns % active neurons per time bin.
    Active = fired ≥1 spike in the bin.
    """
    perc = np.zeros(len(t_bins)-1)
    for k in range(len(perc)):
        t0, t1 = t_bins[k], t_bins[k+1]
        mask = (spike_monitor.t >= t0) & (spike_monitor.t < t1)
        active = np.unique(spike_monitor.i[mask])
        perc[k] = 100 * len(active) / N
    return perc

def select_k_nearest_random(pre_pos, post_pos, k, pool_factor=200):
    """
    Select k presynaptic neurons randomly from the k*pool_factor nearest.
    """
    d2 = np.sum((pre_pos - post_pos) ** 2, axis=1)
    pool = np.argsort(d2)[:k * pool_factor]
    return np.random.choice(pool, k, replace=False)

punished = []

for trial in range(2):
    seed(42)
    np.random.seed(42)
    start_scope()

    ##################################################
    # Parameters
    ##################################################
    Gr_N = 10000
    Go_N = 900
    MF_N = 500

    tau_Gr_MF  = 2.86*ms
    tau_Gr_Go  = 5.0*ms
    tau_Go_MF  = 2.87*ms
    tau_Go_Gr  = 2.86*ms
    tau_Gr_Thr = 1.7*ms
    tau_Go_Thr = 2.0*ms
    tau_PC = 5.0*ms
    tau_adapt = 3.0*ms

    E_leak = -60
    E_ex   = 0
    E_inh  = -80

    g_Gr_leak = 0.07
    g_Go_leak = 0.07

    W_Gr_MF = 0.15
    W_Gr_Go = 0.15
    W_Go_MF = 0.007
    W_Go_Gr = 0.008
    W_Gr_adapt = 0.0

    Thr_Gr     = -40
    Thr_Go     = -35
    Max_Thr_Gr = -35
    Max_Thr_Go = -25

    ##################################################
    # Timing
    ##################################################
    sim_start = -50*ms
    sim_end   = 500*ms
    duration  = sim_end - sim_start
    defaultclock.dt = 1.0*ms

    ##################################################
    # Neuron equations
    ##################################################
    eqs_Gr = '''
    dv/dt = (-g_Gr_leak*(v-E_leak)
             - g_Gr_MF*(v-E_ex)
             - g_Gr_Go*(v-E_inh)
             - g_adapt*(v-E_inh)) / ms : 1
    dtheta/dt = -(theta-Thr_Gr)/tau_Gr_Thr : 1
    dg_Gr_MF/dt = -g_Gr_MF/tau_Gr_MF : 1
    dg_Gr_Go/dt = -g_Gr_Go/tau_Gr_Go : 1
    dg_adapt/dt = -g_adapt / tau_adapt : 1
    '''

    eqs_Go = '''
    dv/dt = (-g_Go_leak*(v-E_leak)
             - g_Go_MF*(v-E_ex)
             - g_Go_Gr*(v-E_ex)) / ms : 1
    dtheta/dt = -(theta-Thr_Go)/tau_Go_Thr : 1
    dg_Go_MF/dt = -g_Go_MF/tau_Go_MF : 1
    dg_Go_Gr/dt = -g_Go_Gr/tau_Go_Gr : 1
    '''

    ##################################################
    # Neuron groups
    ##################################################
    Gr = NeuronGroup(Gr_N, eqs_Gr,
                     threshold='v > theta',
                     reset='v = E_leak; theta = Max_Thr_Gr; g_adapt += W_Gr_adapt*(1 - g_adapt)',
                     method='euler')
    Gr.v = E_leak
    Gr.theta = Thr_Gr

    Go = NeuronGroup(Go_N, eqs_Go,
                     threshold='v > theta',
                     reset='v = E_leak; theta = Max_Thr_Go',
                     refractory=5*ms,
                     method='euler')
    Go.v = E_leak
    Go.theta = Thr_Go

    ##################################################
    # Positions (evenly spaced grids)
    ##################################################
    def grid_positions(N):
        side = int(np.ceil(np.sqrt(N)))
        xs, ys = np.meshgrid(np.linspace(0,1,side), np.linspace(0,1,side))
        return np.column_stack([xs.ravel(), ys.ravel()])[:N]

    Gr_pos = grid_positions(Gr_N)
    Go_pos = grid_positions(Go_N)
    MF_pos = grid_positions(MF_N)

    ##################################################
    # MF Poisson input
    ##################################################
    MF_bg_rate = 50*Hz
    MF_CS_rate = 100*Hz

    bg_idx = np.random.choice(MF_N, int(0.05*MF_N), replace=False)
    cs_idx = np.random.choice(MF_N, int(0.20*MF_N), replace=False)

    steps = int(duration/defaultclock.dt) + 1
    rates = np.zeros((MF_N, steps)) * Hz

    bg_start = int((-45*ms - sim_start)/defaultclock.dt)
    bg_end   = int((0*ms - sim_start)/defaultclock.dt)
    rates[bg_idx, bg_start:bg_end] = MF_bg_rate

    cs_start = int((0*ms - sim_start)/defaultclock.dt)
    cs_end   = int((500*ms - sim_start)/defaultclock.dt)
    rates[cs_idx, cs_start:cs_end] = MF_CS_rate

    MF_rates = TimedArray(rates.T, dt=defaultclock.dt)
    MF = PoissonGroup(MF_N, rates='MF_rates(t,i)')

    ##################################################
    # Synapses (stochastic local fan-in)
    ##################################################
    MF_Go = Synapses(MF, Go, model='w:1',
                     on_pre='g_Go_MF_post += w*(1-g_Go_MF_post)')
    MF_Go.connect(i=[i for j in range(Go_N)
                     for i in select_k_nearest_random(MF_pos, Go_pos[j], 20)],
                  j=[j for j in range(Go_N) for _ in range(20)])
    MF_Go.w = W_Go_MF

    Gr_Go = Synapses(Gr, Go, model='w:1',
                     on_pre='g_Go_Gr_post += w*(1-g_Go_Gr_post)')
    Gr_Go.connect(i=[i for j in range(Go_N)
                     for i in select_k_nearest_random(Gr_pos, Go_pos[j], 100)],
                  j=[j for j in range(Go_N) for _ in range(100)])
    Gr_Go.w = W_Gr_Go

    MF_Gr = Synapses(MF, Gr, model='w:1',
                     on_pre='g_Gr_MF_post += w*(1-g_Gr_MF_post)')
    MF_Gr.connect(i=[i for j in range(Gr_N)
                     for i in select_k_nearest_random(MF_pos, Gr_pos[j], 3)],
                  j=[j for j in range(Gr_N) for _ in range(3)])
    MF_Gr.w = W_Gr_MF

    Go_Gr = Synapses(Go, Gr, model='w:1',
                     on_pre='g_Gr_Go_post += w*(1-g_Gr_Go_post)')
    Go_Gr.connect(i=[i for j in range(Gr_N)
                     for i in select_k_nearest_random(Go_pos, Gr_pos[j], 3)],
                  j=[j for j in range(Gr_N) for _ in range(3)])
    Go_Gr.w = W_Gr_Go

    ##################################################
    # Purkinje cell
    ##################################################
    PC = NeuronGroup(1, 'dv/dt = -v / tau_PC : 1', method='euler')
    Gr_PC = Synapses(Gr, PC, model='w:1', on_pre='v_post += w')
    Gr_PC.connect()
    Gr_PC.w = 1.0

    ##################################################
    # Monitors
    ##################################################
    Gr_spikes = SpikeMonitor(Gr)
    Go_spikes = SpikeMonitor(Go)
    MF_spikes = SpikeMonitor(MF)
    PC_mon = StateMonitor(PC, 'v', record=True)

    ##################################################
    # CF-dependent LTD
    ##################################################
    CF_start = 150*ms - sim_start
    CF_end   = 155*ms - sim_start

    @network_operation(dt=defaultclock.dt)
    def zero_weights():
        global punished
        if trial == 0 and abs(defaultclock.t - CF_end) < defaultclock.dt/2:
            mask = (Gr_spikes.t >= CF_start) & (Gr_spikes.t <= CF_end)
            punished = np.unique(Gr_spikes.i[mask])
            Gr_PC.w[punished] = 0.0
        elif trial == 1:
            Gr_PC.w[punished] = 0.0

    ##################################################
    # Run
    ##################################################
    run(duration)

    ##################################################
    # Plot PC output
    ##################################################
    plt.figure(figsize=(9,4))
    plt.plot(PC_mon.t/ms + float(sim_start/ms), PC_mon.v[0], label=f'Run {trial+1}')
    plt.axvspan((CF_start + sim_start) * 1000, (CF_end + sim_start) * 1000, color='red', alpha=0.2, label='CF window' if trial==0 else None)
    plt.xlabel('Time (ms)')
    plt.ylabel('PC output')
    plt.title('Purkinje Cell Output')
    plt.legend()
    plt.tight_layout()
    plt.show()

    ##################################################
    # Raster plot
    ##################################################
    plt.figure(figsize=(9,4))
    plt.scatter(Gr_spikes.t/ms + float(sim_start/ms), Gr_spikes.i, s=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Granule cell index')
    plt.title(f'Granule Cell Raster Run {trial+1}')
    plt.tight_layout()
    plt.show()

    bin_size = 5 * ms
    t_bins = arange(0 * ms, duration + bin_size, bin_size)
    t_centers = (t_bins[:-1] + t_bins[1:]) / 2

    Gr_active = percent_active(Gr_spikes, t_bins, Gr_N)
    Go_active = percent_active(Go_spikes, t_bins, Go_N)
    MF_active = percent_active(MF_spikes, t_bins, MF_N)

    plt.figure(figsize=(9, 4))
    plt.plot(t_centers / ms + float(sim_start / ms), Gr_active, label='Gr')
    plt.plot(t_centers / ms + float(sim_start / ms), Go_active, label='Go')
    plt.plot(t_centers / ms + float(sim_start / ms), MF_active, label='MF')

    plt.axvspan(
        (CF_start + sim_start) * 1000,
        (CF_end + sim_start) * 1000,
        color='red', alpha=0.2, label='CF window'
    )

    plt.xlabel('Time (ms)')
    plt.ylabel('% active cells')
    plt.title(f'Population activity (Run {trial + 1})')
    plt.legend()
    plt.tight_layout()
    plt.show()
