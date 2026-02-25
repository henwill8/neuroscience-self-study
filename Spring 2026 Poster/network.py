"""
Network class: build units and synapses, run simulation, optionally save checkpoint.
Run from command line or import and call Network(params, rng).run().
"""
from brian2 import *
import numpy as np

from config import get_default_params
from plotting import SimpleResults, plot_all_figures
from utils import (
    pulse_times_train,
    adjacency_indices_within,
    adjacency_indices_between,
    normal_weights,
    save_network_checkpoint,
)


def _build_weight_matrix(syn_EE, syn_EI, syn_IE, syn_II, nExc, nInh):
    """
    Build full weight matrix W (nUnits x nUnits), W[post, pre] = weight from pre to post.
    Unconnected pairs are 0. Returns matrix in same units as synapse weights (stripped to float).
    """
    nUnits = nExc + nInh
    W = np.zeros((nUnits, nUnits))

    def set_weights(syn, varname, post_offset=0, pre_offset=0):
        pre = np.asarray(syn.i)
        post = np.asarray(syn.j)
        w = np.asarray(getattr(syn, varname)[:])
        if hasattr(w, 'unit'):
            w = np.asarray(w / w.unit)
        for k in range(len(w)):
            W[post_offset + post[k], pre_offset + pre[k]] = float(w[k])

    set_weights(syn_EE, 'jEE', 0, 0)           # E -> E
    set_weights(syn_EI, 'jEI', 0, nExc)        # I -> E (pre is inh)
    set_weights(syn_IE, 'jIE', nExc, 0)        # E -> I (post is inh)
    set_weights(syn_II, 'jII', nExc, nExc)     # I -> I
    return W


class Network:
    """
    Builds and runs the CS-US recurrent network: neurons, CS/US input, recurrent synapses,
    monitors; runs simulation; stores pre/post weight matrices; optionally saves checkpoint.
    """

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng
        defaultclock.dt = params['dt']
        self.n_trials_total = params['nTrials'] + (1 if params.get('include_CS_only_trial', False) else 0)

        self._unitsExc = None
        self._unitsInh = None
        self._synapsesEE = None
        self._synapsesEI = None
        self._synapsesIE = None
        self._synapsesII = None
        self._spikeMonExc = None
        self._spikeMonInh = None
        self._stateMonExc = None
        self._stateMonInh = None

    def _build_neurons(self):
        p = self.params
        rng = self.rng

        unitModel = '''
            dv/dt = (gl * (eLeak - v) - iAdapt +
                     sE - sI + sExt) / Cm +
                     noiseSigma * (Cm / gl)**-0.5 * xi: volt (unless refractory)
            diAdapt/dt = -iAdapt / tauAdapt : amp
            dsE/dt = (-sE + uE) / tauFallE : amp
            duE/dt = -uE / tauRiseE : amp
            dsI/dt = (-sI + uI) / tauFallI : amp
            duI/dt = -uI / tauRiseI : amp
            dsExt/dt = (-sExt + uExt) / tauFallE : amp
            duExt/dt = -uExt / tauRiseE : amp
            eLeak : volt
            vReset : volt
            vThresh : volt
            betaAdapt : amp * second
            gl : siemens
            Cm : farad
        '''
        resetCode = 'v = vReset; iAdapt += betaAdapt / tauAdapt'
        threshCode = 'v >= vThresh'

        p['nInh'] = int(p['propInh'] * p['nUnits'])
        p['nExc'] = int(p['nUnits'] - p['nInh'])

        self._unitsExc = NeuronGroup(
            N=p['nExc'],
            model=unitModel,
            method='euler',
            threshold=threshCode,
            reset=resetCode,
            refractory=p['refrExc'],
            clock=defaultclock,
        )
        self._unitsInh = NeuronGroup(
            N=p['nInh'],
            model=unitModel,
            method='euler',
            threshold=threshCode,
            reset=resetCode,
            refractory=p['refrInh'],
            clock=defaultclock,
        )

        mean_beta = p['betaAdaptExc']
        self._unitsExc.v = p['eLeakExc']
        self._unitsExc.vReset = p['vResetExc']
        self._unitsExc.vThresh = p['vThreshExc']
        self._unitsExc.betaAdapt = mean_beta
        self._unitsExc.eLeak = p['eLeakExc']
        self._unitsExc.Cm = p['membraneCapacitanceExc']
        self._unitsExc.gl = p['gLeakExc']

        self._unitsInh.v = p['eLeakInh']
        self._unitsInh.vReset = p['vResetInh']
        self._unitsInh.vThresh = p['vThreshInh']
        self._unitsInh.betaAdapt = p['betaAdaptInh']
        self._unitsInh.eLeak = p['eLeakInh']
        self._unitsInh.Cm = p['membraneCapacitanceInh']
        self._unitsInh.gl = p['gLeakInh']

    def _build_cs_us_input(self):
        p = self.params
        n_trials_total = self.n_trials_total

        nCS = max(1, int(round(p['propCS'] * p['nExc'])))
        nUS = max(1, int(round(p['propUS'] * p['nExc'])))
        nUS = min(nUS, p['nExc'] - nCS)
        cs_neuron_inds = np.arange(0, nCS)
        us_neuron_inds = np.arange(nCS, nCS + nUS)
        p['cs_neuron_inds'] = cs_neuron_inds
        p['us_neuron_inds'] = us_neuron_inds

        trial_starts_s = np.array([tr * float(p['trialPeriod'] / second) for tr in range(n_trials_total)])

        cs_times_s = pulse_times_train(
            trial_starts_s,
            float(p['CS_train_duration'] / second),
            float(p['CS_Hz'] / Hz),
        )
        n_trials_paired = p['nTrials']
        every_n = p.get('cs_only_every_n_trials')
        if every_n is not None and every_n >= 1:
            trial_inds_US = [i for i in range(n_trials_paired) if (i + 1) % every_n != 0]
            trial_starts_for_US = trial_starts_s[trial_inds_US]
        else:
            trial_starts_for_US = trial_starts_s[:n_trials_paired]
        us_times_s = pulse_times_train(
            trial_starts_for_US + float(p['ISI'] / second),
            float(p['US_train_duration'] / second),
            float(p['US_Hz'] / Hz),
        )

        ISI_s = float(p['ISI'] / second)
        cs_only_trial_inds = []
        if every_n is not None and every_n >= 1:
            cs_only_trial_inds.extend([i for i in range(n_trials_paired) if (i + 1) % every_n == 0])
        if p.get('include_CS_only_trial', False) and n_trials_total > n_trials_paired:
            cs_only_trial_inds.append(n_trials_paired)
        p['us_omit_times_s'] = np.array([trial_starts_s[i] + ISI_s for i in cs_only_trial_inds])
        p['trial_starts_s'] = trial_starts_s
        p['trial_duration_s'] = float(p['trialDuration'] / second)
        cs_only_set = set(cs_only_trial_inds)
        p['trial_conditions'] = np.array(['CS' if i in cs_only_set else 'US' for i in range(n_trials_total)])

        cs_indices_src = np.repeat(np.arange(nCS), len(cs_times_s))
        cs_times_expanded = np.tile(cs_times_s, nCS)
        us_indices_src = np.repeat(np.arange(nUS), len(us_times_s))
        us_times_expanded = np.tile(us_times_s, nUS)

        CS_group = SpikeGeneratorGroup(nCS, cs_indices_src, cs_times_expanded * second)
        US_group = SpikeGeneratorGroup(nUS, us_indices_src, us_times_expanded * second)

        syn_CS = Synapses(CS_group, self._unitsExc, on_pre='uExt_post += ' + str(p['spikeInputAmplitude']) + ' * nA')
        syn_CS.connect(i=np.arange(nCS), j=cs_neuron_inds)
        syn_US = Synapses(US_group, self._unitsExc, on_pre='uExt_post += ' + str(p['spikeInputAmplitude']) + ' * nA')
        syn_US.connect(i=np.arange(nUS), j=us_neuron_inds)

    def _build_recurrent_synapses(self):
        p = self.params
        rng = self.rng
        unitsExc = self._unitsExc
        unitsInh = self._unitsInh

        tauRiseEOverMS = p['tauRiseExc'] / ms
        tauRiseIOverMS = p['tauRiseInh'] / ms

        if p.get('use_stdp', False):
            eqs_EE = '''
                jEE : amp
                dapre/dt = -apre / tau_stdp_pre : 1 (event-driven)
                dapost/dt = -apost / tau_stdp_post : 1 (event-driven)
            '''
            on_pre_EE = '''
                uE_post += jEE / tauRiseEOverMS
                apre += 1
                jEE = clip(jEE + A_plus_stdp * apost, w_min_EE, w_max_EE)
            '''
            on_post_EE = '''
                apost += 1
                jEE = clip(jEE - A_minus_stdp * apre, w_min_EE, w_max_EE)
            '''
            synapsesEE = Synapses(
                source=unitsExc, target=unitsExc,
                model=eqs_EE, on_pre=on_pre_EE, on_post=on_post_EE,
            )
            preInds, postInds = adjacency_indices_within(p['nExc'], p['propConnect'], rng)
            synapsesEE.connect(i=preInds, j=postInds)
            synapsesEE.jEE = normal_weights(p['jEE'], len(synapsesEE), p['weightCV'], rng)
            synapsesEE.apre = 0
            synapsesEE.apost = 0
            synapsesEE.namespace.update({
                'tau_stdp_pre': p['tau_stdp_pre'],
                'tau_stdp_post': p['tau_stdp_post'],
                'A_plus_stdp': p['A_plus_stdp'],
                'A_minus_stdp': p['A_minus_stdp'],
                'w_min_EE': p['w_min_EE'],
                'w_max_EE': p['w_max_EE'],
            })
        else:
            synapsesEE = Synapses(
                model='jEE: amp',
                source=unitsExc, target=unitsExc,
                on_pre='uE_post += jEE / tauRiseEOverMS',
            )
            preInds, postInds = adjacency_indices_within(p['nExc'], p['propConnect'], rng)
            synapsesEE.connect(i=preInds, j=postInds)
            synapsesEE.jEE = normal_weights(p['jEE'], len(synapsesEE), p['weightCV'], rng)

        synapsesEI = Synapses(
            model='jEI: amp',
            source=unitsInh, target=unitsExc,
            on_pre='uI_post += jEI / tauRiseIOverMS',
        )
        preInds, postInds = adjacency_indices_between(p['nInh'], p['nExc'], p['propConnect'], rng)
        synapsesEI.connect(i=preInds, j=postInds)
        synapsesEI.jEI = normal_weights(p['jEI'], len(synapsesEI), p['weightCV'], rng)

        synapsesIE = Synapses(
            model='jIE: amp',
            source=unitsExc, target=unitsInh,
            on_pre='uE_post += jIE / tauRiseEOverMS',
        )
        preInds, postInds = adjacency_indices_between(p['nExc'], p['nInh'], p['propConnect'], rng)
        synapsesIE.connect(i=preInds, j=postInds)
        synapsesIE.jIE = normal_weights(p['jIE'], len(synapsesIE), p['weightCV'], rng)

        synapsesII = Synapses(
            model='jII: amp',
            source=unitsInh, target=unitsInh,
            on_pre='uI_post += jII / tauRiseIOverMS',
        )
        preInds, postInds = adjacency_indices_within(p['nInh'], p['propConnect'], rng)
        synapsesII.connect(i=preInds, j=postInds)
        synapsesII.jII = normal_weights(p['jII'], len(synapsesII), p['weightCV'], rng)

        synapsesEE.delay = ((rng.random(synapsesEE.delay.shape[0]) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rng.random(synapsesEI.delay.shape[0]) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rng.random(synapsesIE.delay.shape[0]) * p['delayExc'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rng.random(synapsesII.delay.shape[0]) * p['delayInh'] / defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self._synapsesEE = synapsesEE
        self._synapsesEI = synapsesEI
        self._synapsesIE = synapsesIE
        self._synapsesII = synapsesII

    def _build_monitors(self):
        p = self.params
        self._spikeMonExc = SpikeMonitor(self._unitsExc)
        self._spikeMonInh = SpikeMonitor(self._unitsInh)

        n_rec = p.get('n_record_voltage')
        if n_rec is None:
            record_exc = True
            record_inh = True
        else:
            n_re = min(int(n_rec), p['nExc'])
            n_ri = min(int(n_rec), p['nInh'])
            record_exc = np.linspace(0, p['nExc'] - 1, n_re, dtype=int)
            record_inh = np.linspace(0, p['nInh'] - 1, n_ri, dtype=int)
        p['record_voltage_exc_inds'] = np.arange(p['nExc']) if record_exc is True else np.asarray(record_exc)
        p['record_voltage_inh_inds'] = np.arange(p['nInh']) if record_inh is True else np.asarray(record_inh)
        self._stateMonExc = StateMonitor(self._unitsExc, 'v', record=record_exc)
        self._stateMonInh = StateMonitor(self._unitsInh, 'v', record=record_inh)

    def run(self):
        """
        Build network (neurons, CS/US input, recurrent synapses, monitors), run simulation,
        store pre/post weight matrices, optionally save checkpoint.
        Returns (params, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh).
        """
        self._build_neurons()
        self._build_cs_us_input()
        self._build_recurrent_synapses()

        p = self.params
        p['weight_matrix_pre'] = _build_weight_matrix(
            self._synapsesEE, self._synapsesEI, self._synapsesIE, self._synapsesII,
            p['nExc'], p['nInh'],
        )

        self._build_monitors()

        run(p['duration'], report=p['reportType'], report_period=p['reportPeriod'], profile=p['doProfile'])

        p['weight_matrix_post'] = _build_weight_matrix(
            self._synapsesEE, self._synapsesEI, self._synapsesIE, self._synapsesII,
            p['nExc'], p['nInh'],
        )

        if p.get('save_checkpoint', False):
            save_network_checkpoint(
                p['checkpoint_path'],
                p,
                self._spikeMonExc,
                self._spikeMonInh,
                self._stateMonExc,
                self._stateMonInh,
            )

        return (
            p,
            self._spikeMonExc,
            self._spikeMonInh,
            self._stateMonExc,
            self._stateMonInh,
        )


def main():
    """Default entry point: run network with default params and show all figures."""
    rng_seed = 42
    seed(rng_seed)
    np.random.seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    params = get_default_params()
    net = Network(params, rng)
    params, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh = net.run()

    results = SimpleResults(
        spikeMonExc,
        spikeMonInh,
        stateMonExc,
        stateMonInh,
        params,
    )
    plot_all_figures(results, show=True)


if __name__ == '__main__':
    main()
