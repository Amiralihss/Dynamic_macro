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
    
    def sys_of_eqs_rbc(self, n, n_guess ,k, z, i1, nz, kgrid, zgrid, P, ss: Steady_state, param: Params, firm: Firm, household: Household):
        """
        **Algorithm**:
            - The function first calculates key variables like the interest rate (`irate`), wage (`wage`), consumption (`con`),
                output (`outp`), and investment (`invest`) based on the given `n`, `k`, and `z`.
            - It then computes the capital choice for the next period (`k_p`) using the updated investment value.
            - Using interpolation, the policy function for `n_p` (labor) is computed for each stochastic shock state (`nz`).
            - For each shock state, related variables are calculated (`r_p`, `w_p`, `c_p`), and the expected right-hand side
                of the Euler equation (`q_p`) is determined.
            - The final residual `ee_res` is computed by taking the expected value across stochastic states.
        Args: 
            - `n`: labor choice in period `t` which gets updated.
            - `n_guess`: The initial choice for labor in period `t`
            - `k`: Capital level in period `t`.
            - `z`: Productivity shock in period `t`, following an AR(1) process.
            - `i1`: Current index of the shock state.
            - `nz`: Number of shock states.
            - `kgrid`: capital state space grid.
            - `zgrid`: stochastic space grid folowwing AR(1) process.
            - `P`: Transition matrix of the AR(1) process which is calculated using tauchen's discretization method
        Returns:
            - `ee_res`: Residual of the Euler equation, representing the difference between expected and actual values.
    
        """

        # TODO: this can be done way smarter by separating the first part where we only define the equations from F.O.C 
        # and then do the interpolation. Then we can use the same function to calculate the sys of equation again by using the same function
        ## This function returns the RHS of the Euler equation for given (k,z) and a guess for c

        # Pre-allocation
        n_p = np.zeros((nz, 1))
        r_p = np.zeros((nz, 1))
        w_p = np.zeros((nz, 1))
        c_p = np.zeros((nz, 1))
        q_p = np.zeros((nz, 1))

        # Initializing required parameters
        #alpha = Params().alpha --> alpha has already been initialized by calling the firm
        delta = param.delta
        beta = param.beta
        sigma = param.sigma
        gamma = param.gamma
        chi = ss.chi

        # Initializing agents

        # our n is here the n_t that we want to find so that this sys of equation is solved
        # we start by setting n = n_guess then update it step by step
        irate = firm.mpk(z, k, param, l = n)
        wage = firm.mpl(z, k, n, param)
        # TODO check this equation for mu_c = mu_n what's written here seems to be wrong, Also add chi
        #con = wage / (chi * n ** gamma)#
        # My suggestion : 
        con = (wage/(-household.mu_l_sep(n, param, chi)))#**(1.0/-sigma)
        outp = firm.f(z, k, param, l = n) 
        invest = outp - con

        # given our guess n_guess we find the policy function capital choice for tomorrow k_{t+1}
        k_p = invest + (1.0 - delta) * k

        # we iterate through each stochastic shock to get a policy function n_p (n_t)
        for iz in range(nz):
            n_interp = interpolate.interp1d(
                kgrid, n_guess[:, iz], kind="linear", fill_value="extrapolate"
            )
            # the labor choice corresponds to the capital choice resulting from our sys of equation
            n_p[iz] = n_interp(k_p)
            
            # given this n_p and k_p calculate all the other variables necessary
            r_p[iz] = firm.mpk(zgrid[iz], k_p, param, l = n_p[iz])
            w_p[iz] = firm.mpl(zgrid[iz], k_p, n_p[iz], param) 
            
            # consumption choice from intratemporal labour consumption choice
            c_p[iz] = (w_p[iz] / (chi * n_p[iz] ** gamma))#**(-1.0/param.sigma)

            # the rhs of the Euler equation (1)
            # TODO if possible change this using the mu_c functions that we defined in household
            q_p[iz] = beta * con / c_p[iz]

        # The rhs of the euler equation (calculating expected value) (2)
        ee_sum = P[i1, :] @ (q_p[:] * (r_p[:] + 1.0 - delta))
        # last step of calculating the rhs of the euler equation (3)
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
        kgrid, zgrid, P = grid.setup_grid_TI()
        
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
        **Algorithm**:
            - The loop iteratively adjusts the labor policy `n` over 1,000 iterations or until convergence.
            - At each iteration, for each point in the capital and shock state grid (`kgrid` and `zgrid`), it solves the system of 
                equations using a root-finding method (`fsolve`) to find the optimal `n`.
            - After updating `n`, it calculates a convergence metric based on the maximum absolute difference between the new
                and previous guesses of `n`. If the metric is below the specified tolerance (`1e-5`), the loop exits.

         **Policy Function Computation**
            - Once the main iteration loop converges, the policy functions for consumption (`c_pol`) and capital in the next
                period (`kp_pol`) are computed.
            - For each shock state, the optimal wage, consumption, output, and investment are computed using the final labor 
                policy function (`n_pol`).
            - These variables then determine the policy functions `c_pol` and `kp_pol`.

        Returns:
            - `kgrid` (`numpy.ndarray`): Grid of capital states in the current period.
            - `zgrid` (`numpy.ndarray`): Grid of productivity shock states, following an AR(1) process.
            - `P` (`numpy.ndarray`): Transition matrix for the productivity shocks, representing the probabilities of moving 
                between states in `zgrid`.
            - `kp_pol` (`numpy.ndarray`): Policy function for next period's capital choices `k_{t+1}`, based on the current 
                state variables.
            - `n_pol` (`numpy.ndarray`): Policy function for optimal labor supply choices `n_t`, derived from solving the 
                system of equations.
            - `c_pol` (`numpy.ndarray`): Policy function for optimal consumption choices `c_t`, computed from the intratemporal 
                labor-consumption choice.
            - `wage` (`numpy.ndarray`): Wage variable, calculated as the marginal product of labor for each state in `zgrid` 
                and `kgrid`.
            - `outp` (`numpy.ndarray`): Output variable `Y_t` for each state, computed as a function of `kgrid`, `n_pol`, and `zgrid`.
            - `invest` (`numpy.ndarray`): Investment choices `I_t` for each state, derived as the difference between output and 
                consumption.
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
        kgrid, zgrid, P = grid.setup_grid_TI()
        
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

        for iter0 in range(maxiter):
            # iterate through all the points in the capital grid to find the optimal corresponding labour supply
            for ik in range(nk):
                # since we have uncertainty also iterate through the whole stochastic shock space to get optimal labour supply for each capital and state
                for iz in range(nz):
                    # the optimal labour supply is the root of the sys of equation resulting from the FOCs
                    # start the root finding algorithm with n_guess
                    root = optimize.fsolve(
                        self.sys_of_eqs_rbc, n_guess[ik, iz], args=(n_guess, kgrid[ik], zgrid[iz], iz, nz, kgrid, zgrid, P, ss, param, firm, household)
                    )
                    n_pol[ik, iz] = root

            metric = np.amax(np.abs(n_pol - n_guess))
            print(iter0, metric)
            if metric < tol:
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