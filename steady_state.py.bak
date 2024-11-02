#from params import Params 

class Steady_state:
# TODO use the MPK function and other necessary functions to calculate steady state
    def __init__(self,
                param_instance,
                *args,
                **kwargs):
        
        self.alpha = param_instance.alpha
        self.beta = param_instance.beta
        self.delta = param_instance.delta

    def steady_state_stoch_ram(self):
        base = ((1.0/self.beta)-1.0+self.delta)/self.alpha
        exponent = 1.0/(self.alpha-1.0)
        k_ss = base**exponent
        c_ss = k_ss**(self.alpha)- self.delta*k_ss
    return k_ss, c_ss