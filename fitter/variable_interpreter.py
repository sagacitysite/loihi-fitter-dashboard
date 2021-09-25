from brian2 import Equations
from brian2.units import *

# TODO:
# - catch and re

# - think about if it is usefull to let the user set precision dt? I think it is not usefull at the moment

class VariableInterpreter():
    
    def __init__(self, equation):
        self.equation = equation
        self.identifiers = {}
        self.neuron_namespace = {}
        self.input_variable = False
        self.output_variable ='I'
        self.resting_variable=''
        self.resting_correction = 0
        self.threshold = (0, mV)
        self.weight = (0, mV)
        self.method = 'exact'
        self.runtime = (0, ms)
        self.dt = (0.1, ms)
        self.loihi_threshold_weight_mant = 192 # used for the threshold calculation on Loihi. This weight will cross the threshold
        self.loihi_threshold_weight_exp  = 0

    def get_identifier_names(self):
        try:
            return Equations(self.equation).identifiers
        except Exception as e:
            return e

    def set_identifiers(self, identifiers):
        """
        Set the identifiers as a dictionary.

        identifiers:dictionary with name, value and SI unit
            e.g {'v': (10, mV)}
        """
        
        if self.input_variable == False:
            raise Exception('set_variables() needs to be called first')

        if type(identifiers) != dict:
            raise Exception('Identifiers and their values have to be specified as dictionary')
        self.identifiers = identifiers

        # set the SI unit inside the namespace
        for n in self.identifiers:
            self.neuron_namespace[n] = self.identifiers[n][0]*self.identifiers[n][1]
            
        # shift the resting potential, if one, to zero
        if self.resting_variable != "":
            self.resting_correction = self.neuron_namespace[str(self.resting_variable)] # has SI unit
            self.neuron_namespace[str(self.resting_variable)] = 0*self.identifiers[str(self.resting_variable)][1]
            
        # shift threshold accordingly to match with resting = 0       
        # - ensures to always be right, negative resting corr will be added
        threshold = self.threshold[0]*self.threshold[1] - self.resting_correction
        
        # now restore the tuple, Assuming SI will be correct since output_var and thresh having suitable dimension => thresh and resting as well
        self.threshold = (threshold/self.threshold[1], self.threshold[1])


    def assign_essential_variables(self, input_variable, output_variable, resting_variable, threshold, weight, dt, runtime):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.resting_variable = resting_variable
        self.threshold = threshold
        self.weight = weight
        self.dt = dt
        self.runtime = runtime
        
    def __str__(self):
        
        s = 'equation: '+str(self.equation)+str('\n')
        s+= 'identifiers: '+str(self.identifiers )+str('\n')
        s+= 'neuron_namespace: '+str(self.neuron_namespace )+str('\n')
        s+= 'input_variable: '+str(self.input_variable )+str('\n')
        s+= 'output_variable: '+str(self.output_variable )+str('\n')
        s+= 'resting_variable: '+str(self.resting_variable)+str('\n')
        s+= 'resting_correction: '+str(self.resting_correction )+str('\n')
        s+= 'threshold: '+str(self.threshold )+str('\n')
        s+= 'weight: '+str(self.weight )+str('\n')
        s+= 'method: '+str(self.method )+str('\n')
        s+= 'runtime: '+str(self.runtime )+str('\n')
        s+= 'dt: '+str(self.dt )+str('\n')
        s+= 'loihi_threshold_weight_mant: '+str(self.loihi_threshold_weight_mant )+str('\n')
        s+= 'loihi_threshold_weight_exp: '+str(self.loihi_threshold_weight_exp  )+str('\n')
        
        return s
# useless getter/setter
    def get_eq_names(self):
        return Equations(self.equation).diff_eq_names
    
    def get_equation(self):
        return self.equation
    
    def set_input_variable(self, var):
        self.input_variable = var

    def set_output_variable(self, var):
        self.output_variable = var
        
    def get_output_variable(self):
        return self.output_variable
    
    def get_method(self):
        return self.method
        
    def get_neuron_settings(self):
        return self-neuron_settings