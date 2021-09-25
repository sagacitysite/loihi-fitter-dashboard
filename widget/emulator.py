from brian2 import * 
from brian2_loihi import *

def emulate_neuron(
        decay_v, decay_I, threshold_mant, weight_mant,
        runtime=100, ref_time=1, n_bits=8, weight_exponent=0
    ):

    # Input
    _input = LoihiSpikeGeneratorGroup(1, indices=[0], times=[1], name='input')

    # Neuron
    neuron = LoihiNeuronGroup(
        N=1,
        refractory=ref_time,
        threshold_v_mant=threshold_mant,
        decay_v=decay_v,
        decay_I=decay_I,
        name='neuron'
    )
    # Synapse
    synapse = LoihiSynapses(
        _input,
        neuron,
        w_exp=weight_exponent,
        sign_mode=synapse_sign_mode.EXCITATORY,
        num_weight_bits=n_bits,
        name='synapse'
    )
    synapse.connect()
    synapse.w = weight_mant

    # Monitors
    state_mon_v = LoihiStateMonitor(neuron, 'v')
    state_mon_I = LoihiStateMonitor(neuron, 'I')
    spike_mon = LoihiSpikeMonitor(neuron)

    # Network
    net = LoihiNetwork(neuron, _input, synapse, state_mon_I, state_mon_v, spike_mon, name='network')

    # Run
    net.run(runtime)

    # Get times, current and voltage from simulation
    times = state_mon_I.t/ms
    current = state_mon_I.I[0]
    voltage = state_mon_v.v[0]

    return times, current, voltage
