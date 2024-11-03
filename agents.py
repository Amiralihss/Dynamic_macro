
import numpy as np
from params import Params

class Firm:
    def __init__(self, 
                 labour = False,
                 *args,
                 **kwargs
                 ):

        self.labour = labour
        """
        Initialize the firm as an agent in the economy with a labour flag.

        Args:
            alpha (float): share of capital in production
            labour (bool): Flag indicating whether labour is considered in calculations.
            *args TBA (for instructions look at f function) 
            **kwargs TBA (for instructions look at f function)
        """

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
        #    """Gets a parameter value."""
        #    return self.parameters.get(key)
        
        #def __repr__(self):
        #    """Returns a string representation of the parameters."""
        #    return f"Params({self.parameters})"
    
    def f(self, z, k, param: Params, **kwargs):
        """
        A Cobb-Douglas production function based on stochastic shock 'z' and capital stock `k`, 
        and optionally `l` if `labour` is True.
        Args:
            z (float): Stochastic TFP shock.
            k (float): capital stock.
            **kwargs: Additional keyword arguments:
                - l (float, optional): Only used if `self.labour` is True.
                  hours worked or labour
                  Example usage: `f(z, k, l=3.0)`

                - additional_param (type, optional): Placeholder for future expansion.

        Raises:
            ValueError: If `self.labour` is True but `l` is not provided in `kwargs`.

        Returns:
            float: production value.
        """
        labour = self.labour
        alpha = param.alpha
        if labour:
            # Check for 'l' in kwargs when labour is True
            if 'l' not in kwargs:
                raise ValueError("Parameter 'l' is required when 'labour' is True.")
            l = kwargs['l']
            return np.exp(z)*(k**alpha)*(l**(1.0-alpha)) 
        else:
            return np.exp(z)*k**alpha
    
    def mpk(self, z, k, param: Params, **kwargs):
        """
        Calculates the marginal productivity of capital based on stochastic shock 'z' and capital stock `k`, 
        and optionally `l` if `labour` is True.
        Args:
            z (float): Stochastic TFP shock.
            k (float): capital stock.
            **kwargs: Additional keyword arguments:
                - l (float, optional): Only used if `self.labour` is True.
                  hours worked or labour
                  Example usage: `mpk(z, k, l=3.0)`
                - additional_param (type, optional): Placeholder for future expansion.

        Raises:
            ValueError: If `self.labour` is True but `l` is not provided in `kwargs`.

        Returns:
            float: marginal product of capital
        """
        labour = self.labour
        alpha = param.alpha
        if labour:
            # Check for 'l' in kwargs when labour is True
            if 'l' not in kwargs:
                raise ValueError("Parameter 'l' is required when 'labour' is True.")
            l = kwargs['l']
            return np.exp(z)*alpha*(k**(alpha-1.0))*(l**(1.0-alpha)) 
        else:
            return np.exp(z)*alpha*k**(alpha-1.0)
    
    
    def mpl(self, z, k, l, param: Params):
        """
        Calculates the firms marginal productivity of labour based on stochastic shock 'z' and capital stock `k`, 
        and `l`. This function can only be called if labour is true
        Args:
            z (float): Stochastic TFP shock.
            k (float): capital stock.
            l (float): hours worked or labour. 
        Raises:
            RuntimeError: If labour is False when this method is called.
        
        Returns:
            float: marginal productivity of labour
        """
        # Check the labour flag before proceeding
        labour = self.labour
        alpha = param.alpha
        
        if not labour:
            raise RuntimeError("This method can only be called when 'labour' is True.")

        # Perform the method's intended actions
        return np.exp(z)*(1.0-alpha)*k**(alpha)*l**(-alpha)      
    

class Household:
    # TODO Create a function for budget constraint
    # TODO define chi
    def __init__ (self,
                 labour = False,
                 *args,
                 **kwargs):
        self.labour = labour
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
        #    """Gets a parameter value."""
        #    return self.parameters.get(key)
        
        #def __repr__(self):
        #    """Returns a string representation of the parameters."""
        #    return f"Params({self.parameters})"
        
    def u_sep(self, c, param: Params,**kwargs):
        """
        A 'Constant Relative Risk Aversion' function with separable preferences based on consumption and optionally `l` if `labour` is True.
        Args:
            c (float): consumption.
            **kwargs: Additional keyword arguments:
                - l (float, optional): Only used if `self.labour` is True. (Caution, this has to be called only together with the parameter chi) 
                  hours worked or labour  
                - chi (float, optional): Only used if 'self.labour' is True.
                Example usage: `u(c, l=3.0, chi = 2)`
                - additional_param (type, optional): Placeholder for future expansion.

        Raises:
            ValueError: If `self.labour` is True but `l` and 'chi' is not provided in `kwargs`. 

        Returns:
            float: utility from consumption and disutility from labour.
        """
        sigma = param.sigma
        gamma = param.gamma
        if self.labour:
            # Check for 'l' in kwargs when labour is True
            if 'l' not in kwargs and 'chi' not in kwargs:
                raise ValueError("Parameter 'l' and 'chi' is required when 'labour' is True.")
            l = kwargs['l']
            chi = kwargs['chi']

            if sigma == 1:
                return np.log(c) - chi * l**(1+gamma)/(1+gamma)
            else:
                return (c**(1.0-sigma) - 1.0)/(1.0-sigma) - chi * l**(1.0+gamma)/(1.0+gamma)
        else:
            
            if sigma == 1:
                return np.log(c)
            else:
                return (c**(1.0-sigma) - 1.0)/(1.0-sigma)
    
    def mu_c_sep(self, c, param: Params,**kwargs):
        """
        Calculates the marginal utility from consumption from the CRRA Utility with separable preferences
        Args:
            c (float): consumption level.
            **kwargs: Additional keyword arguments: 
                - additional_param (type, optional): Placeholder for future expansion.

        Returns:
            float: marginal utility from consumption 
        """
        sigma = param.sigma
        
        if sigma == 1:
            return 1.0/c
        else:
            return c**(1.0-sigma)
        
    def mu_l_sep(self, l, param: Params, chi): 
       """
        Calculates marginal utility of labour of a CRRA utility function with separable preferences
        Args:
            l (float): hours worked or labour. 
        Raises:
            RuntimeError: If labour is False when this method is called.
        
        Returns:
            float: Marginal utility of labour.
        """
       if not self.labour:
            raise RuntimeError("This method can only be called when 'labour' is True.")
       gamma = param.gamma
       return -chi*l**(gamma)

    pass
