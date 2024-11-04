import numpy as np
from params import Params 
from grid_data import Grid_data
from agents import Firm, Household
from steady_state import *
from scipy import interpolate,optimize

class Solver:
    def __init__(self):
        """
        Initialize the solver with necessary parameters.
        
        Args:
            parameters (dict): A dictionary of parameters needed for the solver.
        """
        #self.parameters = parameters
    
    # def setup_grid(self):
    #    """
    #    Set up the grid for solving the functional approximation.
    #    """
    #    # Initialize or set up grid parameters here
    #    pass
    
    def sys_of_eqs_stoch_ram(self, c, c_guess, k, z, i1, nz, kgrid, zgrid, P):
        
        """
        This function returns the RHS of the Euler equation from the stochastic ramsey model. for given (k,z) and a guess for c
            - for a pair (k_t, z_t) we guess c_guess_t = h_1(k_t,z_t)
            - get k_{t+1} from Budget constraint where for the pair (k_t, z_t) and c_guess_t
            - Compute c_{t+1} = h_1(k_{t+1}, z_{t+1}) by interpolating guess for h_1 at query point (k_{t+1}, z_{t+1})
            - Compute RHS of the Euler equation by using the transition matrix P(z_{t+1}|z_t) which is calculated using tauchen
            - Goal of getting the RHS of the Euler equation is to find the updated guess for consumption (c_guess_t) given the parameters 
              calculated above the RHS is equal to zero (more on this in Solve_TI_stoch_ram)
        Args:
            c (float): the c_t which will be updated from the RHS of Euler Equation.
            c_guess (float array): the initial guess for c_t
            k (float): the given k_t from the pair (k_t, z_t) in the state space
            z (float): the given z_t from the pair (k_t, z_t) in the state space
            i1 (int): index of the stochastic state that we are finding ourselves in
            nz (int): number of stochastic shocks
            kgrid (float array): our k state space
            zgrid (float array): grid of stochastic shocks
            P(numpy array) : Transition matrix from tauchen 
        Returns:
            ee_res(float): RHS of Euler equation
        """ 

        ## Pre-allocation
        #nz = Grid_data().nz
        #kgrid, zgrid, P = Grid_data().setup_grid_egm()
        
        # Initializing required parameters
        #alpha = Params().alpha --> alpha has already been initialized by calling the firm
        param = Params()
        delta = param.delta
        beta = param.beta

        # Initializing agents
        firm = Firm()
        household = Household()

        c_p = np.zeros((nz,1))
        q_p = np.zeros((nz,1))
        
        # Retrieve k_{t+1} from the budget constraint
        # Calculate k_{t+1} = h2(k_{t}, z_{t}) using guess for consumptions
        k_p = firm.f(z,k,param) + (1.0-delta)*k - c

        # since we are in a stochastic case, we need to find the policy functions for all possible stochastic schocks --> iz in nz
        # e.g. given a bad state consumption is like this otherwise like that
        for iz in range(nz):
            # get c_{t+1} by interpolating guess (c_guess) at K_{t+1} for every z_{t+1}(iz) 
            c_interp = interpolate.interp1d(kgrid, c_guess[:,iz], kind='linear', fill_value='extrapolate')
            
            # c_{t+1} at a certain z_{t+1} --> c_p[iz] 
            c_p[iz] = c_interp(k_p)
            
            ## RHS of Euler without expected value
            # manually putting mu_c for log utility case --> c = c_t and c_p[iz] = c_{t+1} 
            
            # c is the unknown parameter here that we want to estimate by setting the RHS of euler equation to zero to get the updated
            # consumption policy function (c_t = h1(k_t, z_t))
            # using mu_c function from household class
            q_p[iz] = beta*household.mu_c_sep(c_p[iz],param)/household.mu_c_sep(c ,param)*(firm.mpk(zgrid[iz],k_p,param) + 1.0 - delta)
        
        ee_sum = P[i1,:] @ q_p[:]

        ee_res = 1.0 - ee_sum

        return ee_res
    
    def sys_of_eqs_rbc(self, n, n_guess ,k, z, i1, nz, kgrid, zgrid, P):
        ## This function returns the RHS of the Euler equation for given (k,z) and a guess for c

        # Pre-allocation
        n_p = np.zeros((nz, 1))
        r_p = np.zeros((nz, 1))
        w_p = np.zeros((nz, 1))
        c_p = np.zeros((nz, 1))
        q_p = np.zeros((nz, 1))

        # Initializing required parameters
        #alpha = Params().alpha --> alpha has already been initialized by calling the firm
        param = Params(alpha = 0.33,
                     beta = 0.99,
                     delta = 0.025,
                     )
        delta = param.delta
        beta = param.beta
        sigma = param.sigma
        gamma = param.alpha
        ss = Steady_state(param, labour = True)
        chi = ss.chi

        # Initializing agents
        firm = Firm(labour = True)
        household = Household(labour = True)
        irate = firm.mpk(z, k, param, l = n)
        wage = firm.mpl(z, k, n, param)
        
        # TODO check this equation for mu_c = mu_n what's written here seems to be wrong, Also add chi
        
        #con = wage / (chi * n ** gamma)#
        # My suggestion : 
        con = (wage/(-household.mu_l_sep(n, param, chi)))#**(1.0/-sigma)
        outp = firm.f(z, k, param, l = n) 
        invest = outp - con

        k_p = invest + (1.0 - delta) * k

        for iz in range(nz):

            n_interp = interpolate.interp1d(
                kgrid, n_guess[:, iz], kind="linear", fill_value="extrapolate"
            )
            n_p[iz] = n_interp(k_p)
            r_p[iz] = (
                firm.mpk(zgrid[iz], k_p, param, l = n_p[iz])
            )
            w_p[iz] = firm.mpl(np.exp(zgrid[iz]), k_p, n_p[iz], param) 
            c_p[iz] = (w_p[iz] / (chi * n_p[iz] ** gamma))#**(-1.0/param.sigma)
            q_p[iz] = beta * con / c_p[iz]

        ee_sum = P[i1, :] @ (q_p[:] * (r_p[:] + 1.0 - delta))

        ee_res = 1.0 - ee_sum

        return ee_res


    def solve_TI_stoch_ram(self):
        """
        This function solves the stochastic ramsey model using Time Iteration algorithm
            - for all `k` in the `kgrid` space (`kgrid[ik]`)
            - calculate the `c_t` that is the root of the `sys_of_eqs_stoch_ram` given every single stochastic state `z_t` possible (`zgrid[iz]`)
            - the resulting `c_t` (`c_pol`) is going to be our updated `c_guess` if the algorithm doesn't converge and the process goes on
        Returns:
            - kgrid(array) : The capital state space
            - kp_pol(array) : Capital poilcy function for every point in K x Z state space : `k_{t+1}`
            - c_pol(array) : The consumption policy function for every point K x Z state space: `c_t` 
        """

        # TODO: how to make it versatile for other methods
        ## Pre-Allocation
        param = Params()
        ss = Steady_state(param)
        # Initializing Grid
        grid = Grid_data(ss)
        nz = grid.nz
        nk = grid.nk
        kgrid, zgrid, P = grid.setup_grid_egm()
        
        # Initializing required parameters
        alpha = param.alpha
        delta = param.delta
        maxiter = param.maxiter
        tol = param.tol

        # Initializing agents
        firm = Firm()
        #household = Household()
        ## Guess for consumption
        c_guess = np.zeros((nk,nz))
        c_guess[:,:] = ss.c_ss

        # Main Loop
        c_pol = np.zeros((nk,nz))

        for iter0 in range(maxiter):
            # we want to find policy functions for all k in k grid state space
            for ik in range(nk):
                # for all stochastic shocks there needs to be a corresponding policy
                for iz in range(nz):
                    # find c_t s.t. the euler equation is equal to 0 use c_guess as c_t 
                    root = optimize.fsolve(self.sys_of_eqs_stoch_ram, c_guess[ik,iz], args=(c_guess, kgrid[ik], zgrid[iz], iz, nz, kgrid, zgrid, P))
                    # set the c_t from the root finding process as the new guess
                    c_pol[ik,iz] = root

            # evaluate whether update is marginal
            metric = np.amax(np.abs(c_pol-c_guess))
            
            print(iter0,metric)
            
            if (metric<tol):
                break
            # update the guess and continue with root finding
            c_guess[:] = c_pol

        # Compute next period's capital for all grid points

        kp_pol = np.zeros((nk,nz))

        for iz in range(nz):
            # Manually inputing production function in the budget constraint
            #kp_pol[:,iz] = np.exp(zgrid[iz])*kgrid**alpha + (1.0-delta)*kgrid - c_pol[:,iz]
            
            # using the firms production function from class Firm
            kp_pol[:,iz] = firm.f(zgrid[iz], kgrid, param) + (1.0-delta)*kgrid - c_pol[:,iz]

        return kgrid, kp_pol, c_pol


    def solve_TI_rbc(self):

        """
        explain algorithm
        """
        ## Pre-Allocation
        param = Params(alpha = 0.33,
                     beta = 0.99,
                     delta = 0.025,
                     )
        ss = Steady_state(param, labour= True)
        # Initializing Grid
        grid = Grid_data(ss)
        nz = grid.nz
        nk = grid.nk
        kgrid, zgrid, P = grid.setup_grid_egm()
        
        # Initializing required parameters
        alpha = param.alpha
        delta = param.delta
        gamma = param.gamma
        sigma = param.sigma
        chi = ss.chi
        maxiter = param.maxiter
        tol = param.tol

        # Initializing agents
        firm = Firm(labour = True)
        household = Household(labour = True)
        ## Guess for consumption
        n_guess = np.zeros((nk,nz))
        n_guess[:,:] = ss.n_ss
        n_pol = np.zeros((nk, nz))

        for iter0 in range(1000):

            for ik in range(nk):
                for iz in range(nz):
                    root = optimize.fsolve(
                        self.sys_of_eqs_rbc, n_guess[ik, iz], args=(n_guess, kgrid[ik], zgrid[iz], iz, nz, kgrid, zgrid, P)
                    )
                    n_pol[ik, iz] = root

            metric = np.amax(np.abs(n_pol - n_guess))
            print(iter0, metric)
            if metric < 1e-5:
                break

            n_guess[:] = n_pol


# Compute the policy functions for consumption and next period's capital

        c_pol = np.zeros((nk, nz))
        kp_pol = np.zeros((nk, nz))

        for iz in range(nz):

            wage = firm.mpl(zgrid[iz], kgrid, n_pol[:, iz], param)
            # TODO check this equation for mu_c = mu_n what's written here seems to be wrong, Also add chi
            # c_pol[:, iz] = wage / (chi * n_pol[:, iz] ** gamma)
            c_pol[:, iz] = (wage/(-household.mu_l_sep(n_pol[:, iz], param, chi)))#**(1.0/-sigma)
            outp = firm.f(zgrid[iz], kgrid, param, l = n_pol[:, iz]) 
            invest = outp - c_pol[:, iz]
            kp_pol[:, iz] = invest + (1.0 - delta) * kgrid
        
        return kgrid, zgrid, P, kp_pol, n_pol, c_pol, wage, outp, invest


    def solve_neural_network(self):
        """
        Solve the functional approximation using a neural network approach.
        
        Returns:
            result (object): The result from the neural network approach.
        """
        # Implement the neural network approach here
        pass
    
    def compare_methods(self):
        """
        Compare the results of the endogenous grid and neural network methods.
        
        Returns:
            comparison (dict): A dictionary comparing results of both methods.
        """
        # Implement comparison logic here
        pass
    
    def visualize_results(self):
        """
        Visualize the results from different methods.
        """
        # Implement visualization logic here
        pass