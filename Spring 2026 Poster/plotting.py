import numpy as np
from brian2 import *

class SimpleResults:
    def __init__(self, spikeMonExc, spikeMonInh,
                 stateMonExc, stateMonInh,
                 params):

        self.p = params

        # --- Store spike data (convert to seconds)
        self.spikeMonExcT = spikeMonExc.t / second
        self.spikeMonExcI = spikeMonExc.i

        self.spikeMonInhT = spikeMonInh.t / second
        self.spikeMonInhI = spikeMonInh.i

        # --- Store voltages (convert to mV)
        self.stateMonExcV = stateMonExc.v / mV
        self.stateMonInhV = stateMonInh.v / mV

        # --- Time vector for voltage traces
        self.stateDT = float(stateMonExc.clock.dt / second)
        self.duration = float(params['trialTime'] / second)
        self.stateMonT = np.arange(0, self.duration, self.stateDT)

    # -------------------------------------------------
    # Raster plot
    # -------------------------------------------------
    def plot_spike_raster(self, ax):

        ax.scatter(self.spikeMonExcT,
                   self.spikeMonExcI,
                   s=1, c='cyan', marker='.')

        ax.scatter(self.spikeMonInhT,
                   self.p['nExc'] + self.spikeMonInhI,
                   s=1, c='red', marker='.')

        ax.set_xlim(0, self.duration)
        ax.set_ylim(0, self.p['nUnits'])
        ax.set_ylabel("Neuron index")
        ax.set_xlabel("Time (s)")

    # -------------------------------------------------
    # Firing rate
    # -------------------------------------------------
    def plot_firing_rate(self, ax, bin_size=5*ms):

        bin_size = float(bin_size / second)
        bins = np.arange(0, self.duration, bin_size)

        FRExc, _ = np.histogram(self.spikeMonExcT, bins)
        FRInh, _ = np.histogram(self.spikeMonInhT, bins)

        FRExc = FRExc / bin_size / self.p['nExc']
        FRInh = FRInh / bin_size / self.p['nInh']

        centers = bins[:-1] + bin_size/2

        ax.plot(centers, FRExc, color='cyan', alpha=0.6)
        ax.plot(centers, FRInh, color='red', alpha=0.6)

        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")

    # -------------------------------------------------
    # Voltage trace
    # -------------------------------------------------
    def plot_voltage(self, ax, unitType='Exc', neuron_index=0):

        if unitType == 'Exc':
            v = self.stateMonExcV[neuron_index]
            color = 'cyan'
        else:
            v = self.stateMonInhV[neuron_index]
            color = 'red'

        ax.plot(self.stateMonT, v, color=color, lw=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
