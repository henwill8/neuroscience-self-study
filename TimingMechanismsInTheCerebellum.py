from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Helpers
###############################################################################

def gbar_from_psp(psp, E_syn, V_rest, g_l):
    return g_l * abs(psp / (E_syn - V_rest))

seed(42)
np.random.seed(42)
start_scope()

###############################################################################
# Parameters
###############################################################################

Gr_N, Go_N, MF_N, Ba_N, PC_N, Nu_N, CF_N = 10000, 900, 600, 60, 20, 6, 1
V_rest = -65 * mV

###############################################################################
# Synapse parameters
###############################################################################

synapses = {
    ("mossy", "golgi"):      dict(tau=5*ms,  psp=2*mV,   E=0*mV,    n=20),
    ("mossy", "granule"):    dict(tau=75*ms, psp=12*mV,  E=0*mV,    n=(2,6)),
    ("mossy", "nucleus"):    dict(tau=5*ms,  psp=1*mV,   E=0*mV,    n=100),

    ("granule", "golgi"):    dict(tau=2*ms,  psp=2*mV,   E=0*mV,    n=100),
    ("granule", "basket"):   dict(tau=1*ms,  psp=2*mV,   E=0*mV,    n=250),
    ("granule", "purkinje"): dict(tau=5*ms,  psp=2*mV,   E=0*mV,    n=8000),

    ("golgi", "granule"):    dict(tau=50*ms, psp=-6*mV,  E=-75*mV,  n=3),
    ("basket", "purkinje"):  dict(tau=5*ms,  psp=-12*mV, E=-75*mV,  n=10),

    ("purkinje", "nucleus"): dict(tau=8*ms,  psp=-8*mV,  E=-75*mV,  n=15),
    ("nucleus", "climbing"): dict(tau=10*ms, psp=-10*mV, E=-75*mV,  n=6),
    ("climbing", "purkinje"): dict(tau=30 * ms, psp=-20 * mV, E=-75 * mV, n=1),
}

###############################################################################
# Timing
###############################################################################

sim_start = -50 * ms
sim_end = 500 * ms
sim_time = sim_end - sim_start
defaultclock.dt = 1.0 * ms

CS_duration = 200*ms   # total CS duration
CS_phasic_duration = 20*ms
CS_start = 100*ms      # when CS starts
US_start = 250*ms      # when US starts
US_duration = 20*ms    # US pulse duration

###############################################################################
# Neuron parameters
###############################################################################

C_m_val = 200*pF
g_l_val = 10*nS

u0_val     = -50*mV
u_max_val  = -30*mV
tau_u_val  = 10*ms

###############################################################################
# Neuron equations (separate currents for each synapse type)
###############################################################################

# Granule neurons
Gr_eqs = '''
dv/dt = (-g_l*(v - E_l) - I_MF - I_Go) / C_m : volt
du/dt = -(u - u0)/tau_u : volt

# Synaptic currents
I_MF = gbar_MF * g_MF * (v - E_MF) : amp
I_Go = gbar_Go * g_Go * (v - E_Go) : amp

# Conductance dynamics
dg_MF/dt = -g_MF/tau_MF : 1
dg_Go/dt = -g_Go/tau_Go : 1

# Neuron parameters
g_l : siemens
E_l : volt
C_m : farad
u0 : volt
u_max : volt
tau_u : second

# Synapse parameters
tau_MF : second
tau_Go : second
E_MF : volt
E_Go : volt
gbar_MF : siemens
gbar_Go : siemens
'''

# Golgi neurons
Go_eqs = '''
dv/dt = (-g_l*(v - E_l) - I_MF - I_Gr) / C_m : volt
du/dt = -(u - u0)/tau_u : volt

I_MF = gbar_MF * g_MF * (v - E_MF) : amp
I_Gr = gbar_Gr * g_Gr * (v - E_Gr) : amp

dg_MF/dt = -g_MF/tau_MF : 1
dg_Gr/dt = -g_Gr/tau_Gr : 1

g_l : siemens
E_l : volt
C_m : farad
u0 : volt
u_max : volt
tau_u : second

tau_MF : second
tau_Gr : second
E_MF : volt
E_Gr : volt
gbar_MF : siemens
gbar_Gr : siemens
'''

# Basket neurons
Ba_eqs = '''
dv/dt = (-g_l*(v - E_l) - I_Gr) / C_m : volt
du/dt = -(u - u0)/tau_u : volt

I_Gr = gbar_Gr * g_Gr * (v - E_Gr) : amp

dg_Gr/dt = -g_Gr/tau_Gr : 1

g_l : siemens
E_l : volt
C_m : farad
u0 : volt
u_max : volt
tau_u : second

tau_Gr : second
E_Gr : volt
gbar_Gr : siemens
'''

# Purkinje neurons
PC_eqs = '''
dv/dt = (-g_l*(v - E_l) - I_Gr - I_Ba - I_CF) / C_m : volt
du/dt = -(u - u0)/tau_u : volt

I_Gr = gbar_Gr * g_Gr * (v - E_Gr) : amp
I_Ba = gbar_Ba * g_Ba * (v - E_Ba) : amp
I_CF = gbar_CF * g_CF * (v - E_CF) : amp

dg_Gr/dt = -g_Gr/tau_Gr : 1
dg_Ba/dt = -g_Ba/tau_Ba : 1
dg_CF/dt = -g_CF/tau_CF : 1

g_l : siemens
E_l : volt
C_m : farad
u0 : volt
u_max : volt
tau_u : second

tau_Gr : second
tau_Ba : second
tau_CF : second
E_Gr : volt
E_Ba : volt
E_CF : volt
gbar_Gr : siemens
gbar_Ba : siemens
gbar_CF : siemens
'''

# Nucleus neurons
Nu_eqs = '''
dv/dt = (-g_l*(v - E_l) - I_MF - I_PC) / C_m : volt
du/dt = -(u - u0)/tau_u : volt

I_MF = gbar_MF * g_MF * (v - E_MF) : amp
I_PC = gbar_PC * g_PC * (v - E_PC) : amp

dg_MF/dt = -g_MF/tau_MF : 1
dg_PC/dt = -g_PC/tau_PC : 1

g_l : siemens
E_l : volt
C_m : farad
u0 : volt
u_max : volt
tau_u : second

tau_MF : second
tau_PC : second
E_MF : volt
E_PC : volt
gbar_MF : siemens
gbar_PC : siemens
'''

# Mossy fibers
MF_eqs = '''
dv/dt = (-g_l*(v - (E_l)) + D*(xi*sqrt(ms) + mean)) / C_m : volt
du/dt = -(u - u0)/tau_u : volt
g_l : siemens
E_l : volt
C_m : farad
D : amp
u0 : volt
u_max : volt
tau_u : second
mean : 1
'''

# Phasic: 3% of MFs
num_phasic = int(0.03 * MF_N)
phasic_MF_idx = np.random.choice(MF_N, num_phasic, replace=False)

# Tonic: 1% of MFs
num_tonic = int(0.01 * MF_N)
remaining_idx = np.setdiff1d(np.arange(MF_N), phasic_MF_idx)
tonic_MF_idx = np.random.choice(remaining_idx, num_tonic, replace=False)

delta_phasic = 5*mV       # increase in mean for phasic
delta_tonic = 2*mV        # increase in mean for tonic

# @network_operation(dt=defaultclock.dt)
# def apply_CS(t):
#     # Phasic CS: first 20 ms of CS
#     if CS_start <= t < CS_start + CS_phasic_duration:
#         MF.v[phasic_MF_idx] += delta_phasic * defaultclock.dt / ms
#     # Tonic CS: entire CS duration
#     if CS_start <= t < CS_start + CS_duration:
#         MF.v[tonic_MF_idx] += delta_tonic * defaultclock.dt / ms

###############################################################################
# Neuron groups
###############################################################################

MF = NeuronGroup(MF_N, MF_eqs, threshold='v>u', reset='v=E_l; u=u_max', method='euler')
Gr = NeuronGroup(Gr_N, Gr_eqs, threshold='v>u', reset='v=E_l; u=u_max', method='euler')
Go = NeuronGroup(Go_N, Go_eqs, threshold='v>u', reset='v=E_l; u=u_max', method='euler')
Ba = NeuronGroup(Ba_N, Ba_eqs, threshold='v>u', reset='v=E_l; u=u_max', method='euler')
PC = NeuronGroup(PC_N, PC_eqs, threshold='v>u', reset='v=E_l; u=u_max', method='euler')
Nu = NeuronGroup(Nu_N, Nu_eqs, threshold='v>u', reset='v=E_l; u=u_max', method='euler')

# Initialize
for G in [MF, Gr, Go, Ba, PC, Nu]:
    G.v = V_rest
    G.E_l = V_rest
    G.C_m = C_m_val
    G.g_l = g_l_val
    G.u0 = u0_val
    G.u = u0_val
    G.u_max = u_max_val
    G.tau_u = tau_u_val

MF.D = 0.6 * nA
mean_min = 0.05
mean_max = 0.45

MF.mean = np.random.uniform(mean_min, mean_max, size=MF_N)

###############################################################################
# Set synaptic reversal potentials and time constants
###############################################################################

Gr.tau_MF = synapses[("mossy","granule")]["tau"]
Gr.tau_Go = synapses[("golgi","granule")]["tau"]
Gr.E_MF = synapses[("mossy","granule")]["E"]
Gr.E_Go = synapses[("golgi","granule")]["E"]

Go.tau_MF = synapses[("mossy","golgi")]["tau"]
Go.tau_Gr = synapses[("granule","golgi")]["tau"]
Go.E_MF = synapses[("mossy","golgi")]["E"]
Go.E_Gr = synapses[("granule","golgi")]["E"]

Ba.tau_Gr = synapses[("granule","basket")]["tau"]
Ba.E_Gr = synapses[("granule","basket")]["E"]

PC.tau_Gr = synapses[("granule","purkinje")]["tau"]
PC.tau_Ba = synapses[("basket","purkinje")]["tau"]
PC.tau_CF = synapses[("climbing","purkinje")]["tau"]
PC.E_Gr = synapses[("granule","purkinje")]["E"]
PC.E_Ba = synapses[("basket","purkinje")]["E"]
PC.E_CF = synapses[("climbing","purkinje")]["E"]

Nu.tau_MF = synapses[("mossy","nucleus")]["tau"]
Nu.tau_PC = synapses[("purkinje","nucleus")]["tau"]
Nu.E_MF = synapses[("mossy","nucleus")]["E"]
Nu.E_PC = synapses[("purkinje","nucleus")]["E"]

###############################################################################
# Synapse helper function (with conductance dynamics)
###############################################################################

def connect(pre, post, syn_name, post_var):
    # p = synapses[syn_name]
    S = Synapses(pre, post, model='w : 1',
    on_pre=f'{post_var}_post += w * (1 - {post_var}_post)', method='euler')
    S.connect(p=0.01)
    S.w = 1
    return S

###############################################################################
# Connectivity
###############################################################################

S_MF_Gr = connect(MF, Gr, ("mossy","granule"), "g_MF")
S_MF_Go = connect(MF, Go, ("mossy","golgi"), "g_MF")
S_MF_Nu = connect(MF, Nu, ("mossy","nucleus"), "g_MF")

S_Gr_Go = connect(Gr, Go, ("granule","golgi"), "g_Gr")
S_Gr_Ba = connect(Gr, Ba, ("granule","basket"), "g_Gr")
S_Gr_PC = connect(Gr, PC, ("granule","purkinje"), "g_Gr")

S_Go_Gr = connect(Go, Gr, ("golgi","granule"), "g_Go")

S_Ba_PC = connect(Ba, PC, ("basket","purkinje"), "g_Ba")

S_PC_Nu = connect(PC, Nu, ("purkinje","nucleus"), "g_PC")

# # Count how many MF inputs each granule cell receives
# inputs_per_Gr = np.bincount(S_MF_Gr.j, minlength=Gr_N)
#
# plt.figure(figsize=(6,4))
# plt.hist(inputs_per_Gr, bins=50)
# plt.xlabel("Number of MF inputs")
# plt.ylabel("Number of granule cells")
# plt.title("MF → Gr convergence distribution")
# plt.show()
#
# plt.figure(figsize=(6,6))
# plt.scatter(S_MF_Gr.i, S_MF_Gr.j, s=1, alpha=0.3)
#
# plt.xlabel("Mossy fiber index")
# plt.ylabel("Granule cell index")
# plt.title("MF → Granule synapse connectivity")
#
# plt.xlim(0, MF_N)
# plt.ylim(0, Gr_N)
#
# plt.show()

# Climbing fiber loop
CF = PoissonGroup(CF_N, rates=1*Hz)
S_CF_PC = connect(CF, PC, ("climbing","purkinje"), "g_CF")

S_Nu_CF = Synapses(Nu, CF, on_pre='rates_post *= 0.0')
S_Nu_CF.connect()

###############################################################################
# Monitors
###############################################################################

sp_Gr = SpikeMonitor(Gr)
sp_PC = SpikeMonitor(PC)
vm_PC = StateMonitor(Gr, 'v', record=True)
# Monitor mossy fiber spikes
sp_MF = SpikeMonitor(Gr)

###############################################################################
# Run
###############################################################################

run(sim_time)

###############################################################################
# Plot
###############################################################################

plt.figure(figsize=(10,4))
plt.plot(vm_PC.t/ms, vm_PC.v[0]/mV)
plt.xlabel("Time (ms)")
plt.ylabel("Purkinje Vm (mV)")
plt.title("Purkinje cell membrane potential")
plt.show()

# Create a list of spike times for each neuron
spike_times_by_neuron = [sp_MF.t[sp_MF.i == n]/ms for n in range(MF_N)]

# Plot raster
plt.figure(figsize=(10,6))
plt.eventplot(spike_times_by_neuron, colors='black')
plt.xlabel("Time (ms)")
plt.ylabel("Mossy fiber index")
plt.title("Mossy fiber spike raster")
plt.show()