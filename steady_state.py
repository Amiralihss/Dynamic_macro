from params import Params 
from agents import *

class Steady_state:
# TODO use the MPK function and other necessary functions to calculate steady state --> this makes it really difficult
    def __init__(self,
                param_instance: Params,
                labour = False,
                n_ss = 0.66,
                *args,
                **kwargs):
        
        self.n_ss = n_ss
        self.alpha = param_instance.alpha
        self.beta = param_instance.beta
        self.delta = param_instance.delta
        self.labour = labour
        self.z_ss = 1

        if not labour:
            # Steady State of the stochastic ramsey model
            base = ((1.0/self.beta)- 1.0 + self.delta)/self.alpha
            exponent = 1.0/(self.alpha-1.0)
            self.k_ss = base**exponent
            self.c_ss = self.k_ss**(self.alpha)- self.delta*self.k_ss
        else:
            # Steady State of the RBC Model with additively separable utility function
            self.r_ss = 1.0 / self.beta - 1.0 + self.delta
            self.k_ss = (self.r_ss / (self.alpha * self.n_ss ** (1.0 - self.alpha))) ** (1.0 / (self.alpha - 1.0))
            self.i_ss = self.delta * self.k_ss
            self.w_ss = (1.0 - self.alpha) * self.k_ss ** self.alpha * self.n_ss ** (-self.alpha)
            self.y_ss = self.k_ss ** self.alpha * self.n_ss ** (1.0 - self.alpha)
            self.c_ss = self.y_ss - self.i_ss
            self.chi = self.w_ss / (self.c_ss * self.n_ss)    
        