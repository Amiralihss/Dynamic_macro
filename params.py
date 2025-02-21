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

