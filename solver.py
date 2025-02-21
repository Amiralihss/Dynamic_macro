import numpy as np
from params import Params 
from grid_data import Grid_data
from agents import Firm, Household
from steady_state import *
from scipy import interpolate,optimize
from equation_system import EquationSystem
import scipy.sparse as ssp
from scipy.linalg import eig

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
        con = (-household.mu_l_sep(n, param, chi)/wage)**(1.0/-sigma) 
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
        household = Household()
        
        ## Guess for consumption
        c_guess = np.zeros((nk,nz))
        c_guess[:,:] = ss.c_ss

        # Main Loop
        ## Initializing system of equations
        eqs = EquationSystem()
        c_pol = np.zeros((nk,nz))

        for iter0 in range(maxiter):
            # we want to find policy functions for all k in k grid state space
            for ik in range(nk):
                # for all stochastic shocks there needs to be a corresponding policy
                for iz in range(nz):
                    # find c_t s.t. the euler equation is equal to 0 use c_guess as c_t 
                    #root = optimize.fsolve(self.sys_of_eqs_stoch_ram, c_guess[ik,iz], args=(c_guess, kgrid[ik], zgrid[iz], iz, nz, kgrid, zgrid, P))
                    root = optimize.fsolve(eqs.sys_of_eqs_stoch_ram, c_guess[ik,iz], args=(c_guess, kgrid[ik], zgrid[iz], iz, nz, kgrid, zgrid, P, ss, param, firm, household))
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
        ## Initializing system of equations
        eqs = EquationSystem()
        for iter0 in range(maxiter):
            # iterate through all the points in the capital grid to find the optimal corresponding labour supply
            for ik in range(nk):
                # since we have uncertainty also iterate through the whole stochastic shock space to get optimal labour supply for each capital and state
                for iz in range(nz):
                    # the optimal labour supply is the root of the sys of equation resulting from the FOCs
                    # start the root finding algorithm with n_guess
                    root = optimize.fsolve(
                        eqs.sys_of_eqs_rbc, n_guess[ik, iz], args=(n_guess, kgrid[ik], zgrid[iz], iz, nz, kgrid, zgrid, P, ss, param, firm, household)
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

    @staticmethod
    def basefun(grid_x,npx,x):

        vals = np.zeros(2)
        inds = np.zeros(2,dtype=int)

        jl=0
        ju=npx-1

        for iter0 in range(1000):

            if (ju-jl<=1):
                break

            jm=int(np.round((ju+jl)/2))

            if (x>=grid_x[jm]):
                jl=jm
            else:
                ju=jm

        i=jl+1
        vals[1]=(x-grid_x[i-1])/(grid_x[i]-grid_x[i-1])
        vals[0]=( grid_x[i]-x )/(grid_x[i]-grid_x[i-1])
        inds[1]=i
        inds[0]=i-1

        return vals,inds
    @staticmethod
    def aggregation_sparse(afgrid, agrid, a_endo, naf, neta, yprob, na):
        AA = ssp.lil_matrix((naf * neta, naf * neta))
        AA1 = ssp.lil_matrix((naf, naf))
        for j1 in range(neta):
            for i1 in range(naf):
                # When the borrowing constraint is binding:
                if afgrid[i1] < a_endo[0, j1]:
                    AA1[i1, 0] = 1.0
                # When households hit the upper bound:
                elif afgrid[i1] > a_endo[na - 1, j1]:
                    AA1[i1, naf - 1] = 1.0
                else:
                    # Interpolate a' on the finer grid
                    a_endo_int = np.interp(afgrid[i1], a_endo[:, j1], agrid)
                    vals, inds = basefun(afgrid, naf, a_endo_int)
                    AA1[i1, inds[0]] = vals[0]
                    AA1[i1, inds[1]] = vals[1]
            for j2 in range(neta):
                AA[j1 * naf: naf * (j1 + 1), j2 * naf: naf * (j2 + 1)] = AA1 * yprob[j1, j2]
            AA1 = ssp.lil_matrix((naf, naf))
        S, U = ssp.linalg.eigs(AA.T)
        distt = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        distt = distt / np.sum(distt)
        pdf = np.reshape(distt, (neta, naf)).real
        return pdf
    
    def solve_aiyagari_EGM(self):
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
        agrid, zgrid, P = grid.setup_grid_TI()
        
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


        # Find the stationary distribution of the process
        S,U = eig(P.T)
        pistar = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        pistar = pistar / np.sum(pistar)

        # Aggregate labor supply
        labor = zgrid@pistar


        #==============================================================================#
        #                       Computation of equilibrium
        #==============================================================================#

        # Preallocation
        c_guess = np.zeros((na,neta))
        c_new = np.zeros((na,neta))
        a_endo = np.zeros((na,neta))
        a_star = np.zeros(neta)
        con = np.zeros((na,neta))

        pdf = np.zeros((naf,neta))
        cdf = np.zeros((naf,neta))

        # Set bounds for bisection algorithm
        r_bracket = np.array((-delta,(1-betta)/betta))
        # Set a guess for the interest rate
        r_guess = (r_bracket[0]+r_bracket[1])/2

        ## MAIN ALGORITHM ##

        for iter0 in range(maxiter):

            # Set a guess for the interest rate
            irate = r_guess

            # Calculate agg. capital and wage rate through equations (1) and (2)
            cap = ((labor**(alpha-1.0)*(delta + irate))/alpha)**(1.0/(alpha-1.0))
            wage = (1.0-alpha)*cap**alpha*labor**(-alpha)

            # Set a guess for the consumption policy
            for iy in range(neta):
                c_guess[:,iy] = irate*agrid+wage*zgrid[iy]

            # Solve HH problem by EGM
            for iterHH in range(maxiter):

                # Step 1: Determine the grid point(s) where the borrowing constraint is just binding
                for ia in range(na):
                    for iy in range(neta):
                        # Calculate the RHS of euler equation which is the discounted Marginal utility of consumption tomorrow
                        rhs = betta*(1.0+irate)*c_guess[ia,:]**(-gam)@np.transpose(yprob[iy,:])
                        # Calculate the implied consumption (as a function of a')
                        c_new[ia,iy] = rhs**(-1.0/gam)
                        # Use the budget constraint to calculate the implied savings today (as a function of a')
                        a_endo[ia,iy] = (agrid[ia]+c_new[ia,iy]-wage*y[iy])/(1.0+irate)

                        # Calculate savings today that correspond to a choice of a'=0
                        if (ia==0):
                            a_star[iy] = a_endo[0,iy]

                # Step 2: Retrieve an updated guess for the consumption policy
                for ia in range(na):
                    for iy in range(neta):
                        # In this case, use the budget constraint to determine consumption
                        if (agrid[ia]<a_star[iy]):
                            con[ia,iy] = (1.0+irate)*agrid[ia]+wage*y[iy]
                        else:
                            # In this case, use linear interpolation to determine consumption
                            con[ia,iy] = np.interp(agrid[ia],a_endo[:,iy],c_new[:,iy])

                metricEGM = np.amax(np.abs(con-c_guess))

                tolEGM = epsilon*(1.0+np.amax(np.amax(np.abs(c_guess))))

                if (metricEGM<tolEGM):
                    break


                # The new guess for the consumption policy
                c_guess = np.copy(con)
                #c_guess[:] = con

            # Determine the stationary distribution
            pdf = aggregation_sparse(afgrid,agrid,a_endo,naf,neta)

            # Determine the sum of savings held by the household sector
            sav = np.sum(afgrid*np.sum(pdf,0))

            # Update the interest rate guess (Bisection)
            if (cap-sav<0.0):
                r_bracket[1]=irate
            else:
                r_bracket[0]=irate

            # Check stopping criterion:
            tolOuter = epsilon*(1.0 + np.abs(r_bracket[1])+ np.abs(r_bracket[0]))

            print('interest rate guess: ',irate*100,' excess demand for capital: ',cap-sav)

            if (r_bracket[1]-r_bracket[0]<tolOuter or abs(cap-sav)< epsilon):
                break

            r_guess = (r_bracket[0]+r_bracket[1])/2
        return r_guess


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