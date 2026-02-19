"""
A Brian2 helper library for neuron groups with dynamic conductances
and synapse creation.
"""

from brian2 import NeuronGroup, Synapses, ms

class ConductanceNeuronGroup:
    """
    Wrapper for a Brian2 NeuronGroup that allows dynamic addition of
    conductances. Each synapse type gets its own conductance variable
    and decay time constant.
    """

    def __init__(
        self,
        N,
        *,
        g_leak,
        E_leak,
        tau_thr,
        Thr,
        Max_Thr,
        refractory=None,
        method='euler',
    ):
        self.N = N
        self.conductances = {}  # name -> (E_rev, tau)

        # Base equations
        self.base_eqs = '''
dv/dt = (
    - g_leak*(v - E_leak)
    {conductance_terms}
) / ms : 1

dtheta/dt = -(theta - Thr)/tau_thr : 1

g_leak : 1
E_leak : 1
tau_thr : second
Thr : 1
Max_Thr : 1
'''

        self.params = dict(
            g_leak=g_leak,
            E_leak=E_leak,
            tau_thr=tau_thr,
            Thr=Thr,
        )

        self.Max_Thr = Max_Thr
        self.refractory = refractory
        self.method = method

    def add_conductance(self, name, E_rev, tau):
        """
        Register a new synaptic conductance on this neuron group.
        """
        if name not in self.conductances:
            self.conductances[name] = (E_rev, tau)

    def build(self):
        """
        Instantiate the actual Brian2 NeuronGroup with all registered conductances.
        """
        conductance_terms = ''
        conductance_eqs = ''

        for name, (E_rev, tau) in self.conductances.items():
            tau_str = f'{tau/ms}*ms'

            conductance_terms += f'\n    - g_{name}*(v - E_{name})'
            conductance_eqs += f'''
dg_{name}/dt = -g_{name}/({tau_str}) : 1
E_{name} : 1
'''

        eqs = self.base_eqs.format(conductance_terms=conductance_terms) + conductance_eqs
        refractory_str = self._format_refractory(self.refractory)

        self.group = NeuronGroup(
            self.N,
            eqs,
            threshold='v > theta',
            reset='v = E_leak; theta = Max_Thr',
            refractory=refractory_str,
            method=self.method
        )

        # Set scalar parameters
        for k, v in self.params.items():
            setattr(self.group, k, v)

        self.group.Max_Thr = self.Max_Thr

        # Initialize state
        self.group.v = self.params['E_leak']
        self.group.theta = self.params['Thr']

        # Set conductance reversal potentials
        for name, (E_rev, _) in self.conductances.items():
            setattr(self.group, f'E_{name}', E_rev)

        return self.group

    def _format_refractory(self, refractory):
        if refractory is None:
            return False
        if isinstance(refractory, bool):
            return refractory
        if hasattr(refractory, 'dim'):
            return f'{refractory / ms}*ms'
        return refractory

def connect_conductance(
    pre,
    post,
    *,
    name,
    weight,
    connect_kwargs
):
    """
    Create synapses that increment a named conductance on the post neuron.
    """

    S = Synapses(
        pre,
        post,
        model='w : 1',
        on_pre=f'g_{name}_post += w * (1 - g_{name}_post)'
    )

    S.connect(**connect_kwargs)
    S.w = weight
    return S
