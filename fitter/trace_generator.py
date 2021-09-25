#Toughts:
#    is it good to always asume that out_variable.reset = 0? Currently I reset hardcoded to 0! does this make problems with different neuron models?
# the user specifeis an input var, but might be interested in the trace of a different variable. Implement Option to have as many different variable monitored and displayed as the user wants?
# - you have to set the input variable to a certain rest state right ? whats about that?

# - it should be possible to remove the spiking behavior by simply not define it in the neuron subsection!
# - what is about the weights on Loihi? Shall we fit them? Or let the user play arround with it in the tool?
#  - the problem is that one would need to make a weight mapping and therefore would need different weight values, thats maybe an own module/class or a function in LoihiFitter, which just looks arround the weight value?
#from brian2 import NeuronGroup, SpikeGeneratorGroup, Synapses, StateMonitor
#from brian2.units import *
#from brian2.core.network import Network
#from brian2.core.magic import run


# Todo:
# - chekc that rise percet is correctly calculated
from brian2 import *
# Brian use python only
prefs.codegen.target = 'numpy'  # use the Python fallback 'cython'
class TraceGenerator():
    
    def __init__(self, variable_interpreter):
        self.variable_interpreter = variable_interpreter
        self.out_stim_trace       = [] # stores the trace for only setting up the output variable
        #self.out_stim_spike      = False
        self.in_stim_trace        = [] # stores the trace of the ouput variable if input comes trough input var
        self.in_stim_trace_in     = [] # stores the trace of the input variable if input come trough it
        self.threshold_rise       = [] # rise the neron gains trough stimulation
       # self.in_stim_spike    = False
        
    
    def generate_traces(self):
        # get all variables
        neuron_eq       = self.variable_interpreter.equation
        output_var      = self.variable_interpreter.output_variable[0]
        output_var_unit = self.variable_interpreter.output_variable[1]
        input_var       = self.variable_interpreter.input_variable[0]
        input_var_unit  = self.variable_interpreter.input_variable[1]
        threshold       = self.variable_interpreter.threshold[0]
        threshold_unit  = self.variable_interpreter.threshold[1]
        weight          = self.variable_interpreter.weight[0]
        weight_unit     = self.variable_interpreter.weight[1]
        method          = self.variable_interpreter.method
        runtime         = self.variable_interpreter.runtime[0]
        runtime_unit    = self.variable_interpreter.runtime[1]
                    
        # define neuron for only output variable trace and whole input variable stimulation
        neuron0 = NeuronGroup(1,
                            neuron_eq,
                            #threshold=str(output_var) + '>threshold*' + str(threshold_unit),
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            namespace=self.variable_interpreter.neuron_namespace,
                            method = method)
        
        
        neuron1 = NeuronGroup(1,
                            neuron_eq,
                            #threshold=str(output_var) + '>threshold*' + str(threshold_unit),
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            namespace=self.variable_interpreter.neuron_namespace,
                            method = method)
        
        # set initial states
        init_states = {str(output_var): 0*output_var_unit}
        neuron0.set_states(init_states)
        neuron1.set_states(init_states)
        
        # generate input
        indices   = [0]
        times     = [0]*ms
        input_gen = SpikeGeneratorGroup(1, indices, times)

        # connection to output variable
        conn0 = Synapses(input_gen, neuron0, on_pre=str(output_var) + '+= weight*' + str(output_var_unit))
        conn0.connect()
        
        # connection to input variable
        conn1 = Synapses(input_gen, neuron1, on_pre=str(input_var) + '+= weight*' + str(input_var_unit))
        conn1.connect()

        # probe's
        outDecay_mon    = StateMonitor(neuron0, output_var, record=0)
        #outSpike_mon    = SpikeMonitor(neuron0)
        
        inDecay_out_mon = StateMonitor(neuron1, output_var, record=0)
        inDecay_in_mon  = StateMonitor(neuron1, input_var, record=0)
        #inSpike_mon     = SpikeMonitor(neuron1)

        # run
        defaultclock.dt = self.variable_interpreter.dt[0]*self.variable_interpreter.dt[1]
        run_output = run(runtime*runtime_unit, report='text')
        

        # save traces
        # make sure they are flattened, not nested arrays
        self.out_stim_trace     = outDecay_mon.get_states()[output_var].flatten()
        self.in_stim_trace      = inDecay_out_mon.get_states()[output_var].flatten()
        self.in_stim_trace_in   = inDecay_in_mon.get_states()[input_var].flatten()
        
        # check for spike's
        #if outSpike_mon.count[0] > 0:
        #    self.out_stim_spike = True
        #if inSpike_mon.count[0] > 0:
        #    self.in_stim_spike = True
        
        # set rise percent to threshold
        # make sure that units are taken into account
        self.threshold_rise =  (np.max(self.in_stim_trace) / self.variable_interpreter.threshold[1]) / self.variable_interpreter.threshold[0]
        # self.threshold_rise =  np.max(self.in_stim_trace) / (self.variable_interpreter.threshold[0]*self.variable_interpreter.threshold[1])
        return run_output

    def get_trace(self):
        return (self.in_stim_trace / self.variable_interpreter.threshold[1])/self.variable_interpreter.threshold[0]
        
    def get_out_stim_trace(self):
        return self.out_stim_trace
    
    def get_out_stim_trace_unitless(self):
        return self.out_stim_trace / self.variable_interpreter.output_variable[1]
    
    def get_in_stim_trace_unitless(self):
        return self.in_stim_trace / self.variable_interpreter.output_variable[1]
        
    def get_in_stim_trace_in_unitless(self):
        return self.in_stim_trace_in / self.variable_interpreter.input_variable[1]
        
    def get_variable_interpreter(self):
        return self.variable_interpreter
    
    def get_simulation_precision(self):
        return self.variable_interpreter.dt[0]*self.variable_interpreter.dt[1]
    
    def get_simulation_steps(self):
        return self.variable_interpreter.runtime[0]/self.variable_interpreter.dt[0]