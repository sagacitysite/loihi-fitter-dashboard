#from brian2 import Network, ExplicitStateUpdater, StateUpdateMethod, defaultclock, ms, NeuronGroup, SpikeGeneratorGroup, Synapses, StateMonitor, Network, ExplicitStateUpdater, StateUpdateMethod, defaultclock
#from brian2.units import *

import concurrent.futures
from brian2 import *
prefs.codegen.target = 'numpy'  # use the Python fallback 'cython'

import numpy as np

class LoihiFitter():
    # Todo:
    # - can make a function compute_trace(v_decay, I_decay, weight. weight_exp): that computes traces of interest then use it inside this class for find_tresh_weight and compute_final_result_trace()
    
    # notes on find weights:
    #In the end of the day the user has to pick the weight wisely.
    #Since if his weight did a 10% rise and he pikcs a threshold with a weight fo 255 * 2^(6+6) then he is on the final limit of that #EXP and thus will not have a good distriubution on his weights!, Maybe we can autonmate this picking/selecting process and #highlight good values in the table.


    #-----------------------------------------
    #     make this one to
    #-----------------------------------------
    #another idea is to rpove a full range weight on top of the above table.
    #this is only one value and assumes that you use Loihis abillity of weight exp prototype maps.
    #this way the range for the weight spans across all weights possible on Loihi! 
    
    def __init__(self, trace_generator):
        self.trace_generator      = trace_generator
        self.variable_interpreter = trace_generator.variable_interpreter
        self.stimulation_weight   = 255*2**6
        self.v_traces_v           = []
        self.v_traces_I           = []
        self.v_decay              = False
        self.I_decay              = False
        self.threshold_table      = []
        self.threshold_mant       = False
        self.weight_mant          = False
        self.weight_exp           = False
        
        # Define first order forward euler, if not already defined
        if ('forward_euler' not in StateUpdateMethod.stateupdaters):
            eq_forward_euler = '''
            x_new = x + dt * f(x,t)
            '''
            forward_euler = ExplicitStateUpdater(eq_forward_euler, stochastic='none')
            StateUpdateMethod.register('forward_euler', forward_euler)
            
        self.loihi_model = '''
            rnd_v = int(sign(v)*ceil(abs(v*tau_v))) : 1
            rnd_I = int(sign(I)*ceil(abs(I*tau_I))) : 1
            dv/dt = -rnd_v/ms + I/ms: 1
            dI/dt = -rnd_I/ms: 1
            tau_v : 1
            tau_I : 1
            
        '''
        
        
    def fit_model(self):
        # get output_variable decay traces + normalise traces
        # fit v_decay -> output_variable decay trace
        # fit c_decay -> output_variable decay trace inDecay (using found v_decay)
        # find weight with rise value (make an option to choose w_exp)
        # find the right mantissa
        
        v_decay_alignment = normalize(self.trace_generator.get_out_stim_trace_unitless())
        I_decay_alignment = normalize(self.trace_generator.get_in_stim_trace_unitless())
        
        #-------------------------------------------------
        #                  fit v_decay
        #-------------------------------------------------
        net = Network()
        
        # define neurons
        # note due to fast brian simulation and initialisation times it is just faster to simulate all possibilites
        # instead of exploit the convex error function.
        neurons = NeuronGroup(4097,
                            self.loihi_model,
                            #threshold='v>inf',
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            method = 'forward_euler')
        
        # set all parameters
        neurons.tau_I = 0
        neurons.tau_v = '(i*1.0)/2**12' # set all possible tau values the fast way
     
        # define input
        indices   = [0]
        times     = [0]*ms
        input_gen = SpikeGeneratorGroup(1, indices, times)
        
        # connection to v variable
        conn = Synapses(input_gen, neurons, on_pre='v += '+ str(self.stimulation_weight))
        conn.connect()
        
        # monitor all traces
        v_mon = StateMonitor(neurons, 'v', record=True)
        
        # run simulation
        net.add(neurons)
        net.add(input_gen)
        net.add(conn)
        net.add(v_mon)
        defaultclock.dt = 1*ms
        net.run(self.trace_generator.get_simulation_steps()*ms)
        
        # save all traces
        self.v_traces_v = v_mon.v
        
        # find best match
        best_mse = inf
        best_decay = 0
        for n in np.arange(0,4097):
            mse = self.L_n(v_decay_alignment, normalize(self.v_traces_v[n]), 2)
            if mse < best_mse:
                best_mse = mse
                best_decay = n
        self.v_decay =  best_decay
    
        #-------------------------------------------------
        #      fit I_decay with found v_decay
        #-------------------------------------------------
        # define neurons
        neurons = NeuronGroup(4097,
                            self.loihi_model,
                            #threshold='v>inf',
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            method = 'forward_euler')
        
        # set all parameters
        neurons.tau_v = self.v_decay/2**12 # decay is inversed, normally it is v/(2**12/decay)
        neurons.tau_I = '(i*1.0)/2**12'

  
        
        # define input
        indices   = [0]
        times     = [0]*ms
        input_gen = SpikeGeneratorGroup(1, indices, times)
        
        # connection to v variable
        conn = Synapses(input_gen, neurons, on_pre='I += '+ str(self.stimulation_weight))
        conn.connect()
        
        # monitor all traces
        v_mon = StateMonitor(neurons, 'v', record=True)
        
        # run simulation
        net = Network(neurons,
                      input_gen,
                      conn,
                      v_mon)
        defaultclock.dt = 1*ms
        net.run(self.trace_generator.get_simulation_steps()*ms)
        
        # save all traces
        self.v_traces_I = v_mon.v
        
        # find best match
        best_mse = inf
        best_decay = 0
        for n in np.arange(0,4097):
            mse = self.L_n(I_decay_alignment, normalize(self.v_traces_I[n]), 2)
            if mse < best_mse:
                best_mse = mse
                best_decay = n
        self.I_decay =  best_decay
        
    def fit_thres_weight(self):
        # get values
        thres_rise   = self.trace_generator.threshold_rise
        w_thres_mant = self.variable_interpreter.loihi_threshold_weight_mant
        w_thres_exp  = self.variable_interpreter.loihi_threshold_weight_exp 
        
        #-------------------------------------------------
        #             compute trace
        #-------------------------------------------------
        # define neuron
        neuron = NeuronGroup(1,
                            self.loihi_model,
                            #threshold='v>inf',
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            method = 'forward_euler')
        
        # set all parameters
        neuron.tau_v = self.v_decay/2**12
        neuron.tau_I = self.I_decay/2**12
        
        # define input
        indices   = [0]
        times     = [0]*ms
        input_gen = SpikeGeneratorGroup(1, indices, times)
        
        # connection to I variable with the Loihi weight
        # calculate weight
        n_bits                    = 8
        is_mixed                  = 0
        numLsbBits                = 8 - n_bits - is_mixed

        # Shift mantissa
        w_thres_mant = int(np.floor((w_thres_mant / 2**numLsbBits))) * 2**numLsbBits
        # Scale with weight exponent
        w = w_thres_mant * 2 **(6.0+w_thres_exp)
        # Shift scaled weight
        w = int(np.floor(w / 2**6)) * 2**6
        # Apply 21 bit limit
        w = np.clip(w, -2097088, 2097088)
        
        # connect
        conn = Synapses(input_gen, neuron, on_pre='I += ' + str(w))
        conn.connect()
        
        # monitor the trace
        v_mon = StateMonitor(neuron, 'v', record=True)
        
        # run simulation
        net = Network(neuron,
                      input_gen,
                      conn,
                      v_mon)
        defaultclock.dt = 1*ms
        net.run(self.trace_generator.get_simulation_steps()*ms)
        
        #-------------------------------------------------
        #             compute threshold and weight
        #-------------------------------------------------        
        threshold_mant  = int(np.max(v_mon.v[0])/2**6)
        weight_mant     = int(thres_rise * w_thres_mant)
        
        # TODO remove
        self.fit_thres_weight_trace = v_mon.v[0]
        
        self.threshold_mant = threshold_mant
        self.weight_mant    = weight_mant
        self.weight_exp     = w_thres_exp

        
    def fit_mantissa_table(self):        
        # simulate every possible weight
        #-------------------------------------------------
        #             compute weights
        #-------------------------------------------------
        n_bits                    = 8
        is_mixed                  = 0

        numLsbBits                = 8 - n_bits - is_mixed
        min_possible_weight_mant  = 0
        max_possible_weight_mant  = 255
        weight_mantissas          = np.matrix(np.arange(min_possible_weight_mant,
                                              max_possible_weight_mant + 1))
        weight_exp                = np.arange(-8,8)
        weight_mantissas          = np.repeat(weight_mantissas, weight_exp.size, axis=0)

        # Shift weight mantissa
        weight_mantissas = np.floor((weight_mantissas / 2**numLsbBits)).astype(int) * 2**numLsbBits

        # scale for all exponents
        for n, exp in zip(range(weight_exp.size), weight_exp):
            # Scale weight with weight exponent
            weight_mantissas[n] = weight_mantissas[n] * 2 **(6.0+exp)
            # Shift scaled weight
            weight_mantissas[n] = (weight_mantissas[n] / 2**6).astype(int) * 2**6
            # Apply 21 bit limit
            weight_mantissas[n] = np.clip(weight_mantissas[n], -2097088, 2097088)
        weights = weight_mantissas
            
        #-------------------------------------------------
        #             simulate weights
        #-------------------------------------------------
        # define neurons
        neurons = NeuronGroup(weights.size,
                            self.loihi_model,
                            #threshold='v>inf',
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            method = 'forward_euler')
        
        # set all parameters
        neurons.tau_v = self.v_decay/2**12 # decay is inversed, normally it is v/(2**12/decay)
        neurons.tau_I = self.I_decay/2**12
        
        #neurons.tau_I = '(i*1.0)/2**12'

        # define input
        indices   = [0]
        times     = [0]*ms
        input_gen = SpikeGeneratorGroup(1, indices, times)
        
        # connection to v variable
        conn = Synapses(input_gen, neurons, on_pre='I += w', model='w : 1')
        conn.connect()
        conn.w = weights.flatten()
        
        # monitor all traces
        v_mon = StateMonitor(neurons, 'v', record=True)
        
        # run simulation
        net = Network(neurons,
                      input_gen,
                      conn,
                      v_mon)
        defaultclock.dt = 1*ms
        net.run(self.trace_generator.get_simulation_steps()*ms)
        
        # find the voltage higest value for every weight
        maxes =[]
        for n in range(v_mon.v.shape[0]):
            maxes.append(np.max(v_mon.v[n])) 
        maxes = np.array(maxes)
        
        # how many times the input has to be multiplied to get a spike?
        rise_to_100 = 1/self.trace_generator.threshold_rise

        # for every weight/rise compute the threshold
        # threshold = max_value * rise_to_100
        maxes = maxes * rise_to_100
        
        # Loihi uses internal thres_mantissa scaling of 2**6 i.e. threshold = thresh_mantissa * 2**6
        maxes = maxes / 2**6
        
        # clip the ones not possible i.e. mantissa > 131071
        maxes = np.clip(maxes, 0, 131071)
        
        #return maxes
        
        # draw a table with the found weights
        # make weight table 2d and save it
        self.threshold_table = np.array(np.split(maxes, 16)).astype(int)

        
    def print_threshold_table(self):
        if len(self.threshold_table) == 0:
            print('no data, call fit_mantissa_table() first')
            return
        
        mantissa = np.arange(0, 255 + 1)
        actual   = self.threshold_table
        
        # Print header
        print(' '.join(['{:7}' for i in range(17)]).format(*np.insert(np.arange(-8,8), 0, 0)))
        # Print separation line
        print(''.join(['-' for i in range(8*17)]))
        # Print one row for every mantissa, iterate over columns (weight exponents)
        for i in range(actual.shape[1]):
            print(' '.join(['{:7}' for i in range(17)]).format(*np.insert(actual[:,i], 0, mantissa[i])))            
        
    def L_n(self, val0, val1, n):
        val0 = np.array(val0)
        val1 = np.array(val1)

        if(np.size(val0) != np.size(val1)):
            raise ValueError('L_n input vectors must have the same length.')  

        return np.sum(np.abs(val0-val1)**n)/np.size(val0)

    def get_fit_result(self):
        
        if self.v_decay == False:
            raise Exception('Currently no fitted values. Call fit_model() to start fitting.')
            
        d = {}
        d['v_decay'] = self.v_decay
        d['I_decay'] = self.I_decay
        d['threshold_mant'] = self.threshold_mant
        d['weight_mant'] = self.weight_mant
        d['weight_exp'] = self.weight_exp
        
        return d
    
    def __str__(self):
        
        if self.v_decay == False:
            return 'Currently no fitted values. Call fit_model() to start fitting.'
        
        s = 'v_decay: '+str(self.v_decay)+str('\n')
        s+= 'I_decay: '+str(self.I_decay)+str('\n')
        s+= 'threshold_mant: '+str(self.threshold_mant)+str('\n')
        s+= 'weight_mant: '+str(self.weight_mant)+str('\n')
        s+= 'weight_exp: '+str(self.weight_exp)
        
        return s
    
    def compute_trace(self, v_decay, I_decay, weight, weight_exp):
        # define neuron
        neuron = NeuronGroup(1,
                            self.loihi_model,
                            #threshold='v>inf',
                            #reset=str(output_var) + '=0*' + str(output_var_unit),
                            #refractory = 1*ms,
                            method = 'forward_euler')
        
        # set all parameters
        neuron.tau_v = v_decay/2**12
        neuron.tau_I = I_decay/2**12
        
        # define input
        indices   = [0]
        times     = [0]*ms
        input_gen = SpikeGeneratorGroup(1, indices, times)
        
        # connection to I variable with the Loihi weight
        # calculate weight
        n_bits                    = 8
        is_mixed                  = 0
        numLsbBits                = 8 - n_bits - is_mixed

        # Shift mantissa
        weight = int(np.floor((weight / 2**numLsbBits))) * 2**numLsbBits
        # Scale with weight exponent
        w = weight * 2 **(6.0+weight_exp)
        # Shift scaled weight
        w = int(np.floor(w / 2**6)) * 2**6
        # Apply 21 bit limit
        w = np.clip(w, -2097088, 2097088)
        
        # connect
        conn = Synapses(input_gen, neuron, on_pre='I += ' + str(w))
        conn.connect()
        
        # monitor the trace
        v_mon = StateMonitor(neuron, 'v', record=True)
        
        # run simulation
        net = Network(neuron,
                      input_gen,
                      conn,
                      v_mon)
        defaultclock.dt = 1*ms
        net.run(self.trace_generator.get_simulation_steps()*ms)
        
        return v_mon.v[0]

        
        
    def compute_final_result_trace(self):
        if self.v_decay == False:
            raise Exception('nothing fitted yet. Call fit_model() first.')
        return self.compute_trace(self.v_decay, self.I_decay, self.weight_mant, self.weight_exp)

    def get_trace(self):     
        return self.compute_final_result_trace() / (self.threshold_mant*2**6)
    
# Make this class intern  
def normalize(data):
    normal = np.array(data)    
    return normal / np.max(normal)
