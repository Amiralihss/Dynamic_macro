import tauchen as tc
import numpy as np
from params import Params 
from steady_state import Steady_state 

class Grid_data:
    #def __init__(self,
    #             max_T = 32,
    #             batch_size = 8):
    #    self.max_T = max_T
    #    self.batch_size = batch_size
    #    self.time_range = torch.arange(0.0, self.max_T , 1.0)
    #    self.grid = self.time_range.unsqueeze(dim = 1)

    def __init__(self,
                 ss_instance: Steady_state,
                 nk = 21, 
                 nz = 11,
                 theta = 3
                 ):
        
        self.nk = nk
        self.k_min = 0.75*ss_instance.k_ss
        self.k_max = 1.25*ss_instance.k_ss
        self.nz = nz
        self.theta = theta

        
    #kgrid = np.linspace(self.k_min,self.k_max,self.nk)
    #    zgrid,P = tc.approx_markov(Params().rho_z, Params().sig_e, 2, self.nz)

    def setup_grid_TI(self):  
        k_min = self.k_min 
        k_max = self.k_max 
        nk = self.nk 
        nz = self.nz   
        
        kgrid = np.linspace(k_min, k_max, nk)
        zgrid,P = tc.approx_markov(Params().rho_z, Params().sig_e, 2, nz)
        
        return kgrid, zgrid, P
    
    
    def setup_grid_EGM(self):
        gridmin = self.k_min
        gridmax = self.k_max
        nk = self.nk
        nz = self.nz
        theta = self.theta
        # First non equidistant grid for assets
        grid = np.zeros(nk)
        grid[0] = gridmin

        for ig in range(nk-1):
            grid[ig+1]=((ig+1)/(nk-1))**theta*(gridmax-gridmin)+gridmin
        # Discretize the AR-process into a Markov chain
        y,yprob=tc.approx_markov(Params().rho_z, Params().sig_e,3,nz)
        # Normalization
        y=np.exp(y)
        return grid, y, yprob
    
    def setup_grid_nn (self):
        pass

