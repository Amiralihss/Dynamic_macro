"""
params.py

This module contains the Params class where we define the model parameters.
"""

class Params:
    """
    A class to define and store model parameters.
    """
    def __init__(self,
                 alpha = 1.0/3.0,
                 beta = 0.9,
                 delta = 0.1,
                 k_0 = 1.0,
                 rho_z = 0.979,
                 sig_e = 0.007,
                 sigma = 1.0,
                 gamma = 1.0,
                 maxiter = 10000,
                 tol = 1e-5,
                 *args,
                 **kwargs
                ):
        """
        Initializes the Params object with the parameters of an RBC model.

        :param alpha: Share of labour in production.
        :param beta: discount factor.
        :param delta: depreciation rate.
        :param k_0: initial capital stock.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments representing parameter names and values.
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.k_0 = k_0
        self.rho_z = rho_z
        self.sig_e = sig_e
        self.sigma = sigma
        self.gamma = gamma
        self.maxiter = maxiter
        self.tol = tol

        #self.parameters = {
        #    'alpha'  : alpha,
        #    'beta'   : beta,
        #    'delta'  : delta,
        #    'k_0'    : k_0,
        #    'rho_Z'  : rho_z,
        #    'sig_e'  : sig_e,
        #    'sigma'  : sigma,
        #    'gamma'  : gamma,
        #    'maxiter': maxiter,
        #    'tol'    : tol
        #}


        # If args are passed, we can use them and save them as aditional parameters
        #if args:
        #    self.parameters['aditional_params'] = args

        # Update with additional parameters from kwargs
        #self.parameters.update(kwargs)
        
        # Process keyword arguments
        # for the case where we don't have any predefine arguments
        # for key, value in kwargs.items():
        #    self.parameters[key] = value

        #def set_param(self, key, value):
        #    """Sets a parameter value."""
        #    self.parameters[key] = value

        #def get_param(self, key):
        #   """Gets a parameter value."""
        #    return self.parameters.get(key)
        
        #def __repr__(self):
        #    """Returns a string representation of the parameters."""
        #    return f"Params({self.parameters})"
