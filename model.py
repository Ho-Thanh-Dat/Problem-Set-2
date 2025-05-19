"""

model.py
--------
This code sets up the model.

"""

#%% Imports from Python
from numpy import count_nonzero,exp,expand_dims,linspace,log,tile
from scipy import stats
from types import SimpleNamespace

#%% Deterministic Life Cycle Model.
class household():
    '''
    
    Methods:
        __init__(self,**kwargs) -> Set the household's attributes.
        setup(self,**kwargs) -> Sets parameters.
        
    '''
    
    #%% Constructor.
    def __init__(self,**kwargs):
        '''        
        
        This initializes the model.
        
        Optional kwargs:
            All parameters changed by setting kwarg.
            
        '''

        print('--------------------------------------------------------------------------------------------------')
        print('Model')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The model is the deterministic life cycle model and is solved via Backward Induction.')
        
        print('\n--------------------------------------------------------------------------------------------------')
        print('Household')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The household is fintely-lived and dies after T periods.')
        print('   It derives utility from consumption.')
        print('    -> He/she retires after tr periods.')
        
    #%% Set up model.
    def setup(self,**kwargs):
        '''
        
        This sets the parameters and creates the grids for the model.
        
            Input:
                self : Model class.
                kwargs : Values for parameters if not using the default.
                
        '''
        
        # Namespace for parameters, grids, and utility function.
        setattr(self,'par',SimpleNamespace())
        par = self.par

        print('\n--------------------------------------------------------------------------------')
        print('Parameters:')
        print('--------------------------------------------------------------------------------\n')
        
        # Preferences.
        par.beta = 0.99 # Discount factor.
        par.sigma = 5.0 # CRRA.

        # Demographics.
        par.T = 61 # Last period of life.
        par.tr = 41 # First period of retirement.

        # Prices and Income.
        par.r = 0.03 # Interest rate.
        par.kappa = 0.6 # Share of income as pension.

        par.sigma_eps = 0.07 # Std. dev of productivity shocks.
        par.rho = 0.85 # Persistence of AR(1) process.
        par.mu = 0.0 # Intercept of AR(1) process.

        par.ylen = 7 # Grid size for y.
        par.m = 3 # Scaling parameter for Tauchen.
            
        # Simulation parameters.
        par.seed_sim = 2025 # Seed for simulation.
        par.TT = 61 # Number of time periods.
        par.NN = 10000 # Number of time periods.

        # Set up asset grid.
        par.alen = 300 # Grid size for a.
        par.amax = 1000.0 # Upper bound for a.
        par.amin = 0.0 # Minimum a.

        # Set up hour grid
        par.hlen = 300 # Number of hours worked.
        par.hmax = 1.0 # Upper bound for hours worked.
        par.hmin = 0.01 # Minimum hours worked.
        par.hours = linspace(par.hmin,par.hmax,par.hlen) # Equally spaced, linear grid for hours worked.

        # Add in hours parameter
        par.phi = 0.5
        par.eta = 2
        
        # Update parameter values to kwarg values if you don't want the default values.
        for key,val in kwargs.items():
            setattr(par,key,val)
        
        assert par.beta > 0.0 and par.beta < 1.0
        assert par.sigma >= 1.00
        assert par.sigma_eps > 0.00
        assert abs(par.sigma_eps) < 1.00
        assert par.kappa >= 0.0 and par.kappa <= 1.0
        assert par.alen > 5
        assert par.amax > par.amin
        
        # Set up asset grid.
        par.agrid = linspace(par.amin,par.amax,par.alen) # Equally spaced, linear grid for a (and a').

        # Discretize productivity.
        ygrid,pmat = tauchen(par.mu,par.rho,par.sigma_eps,par.ylen,par.m) # Tauchen's Method to discretize the AR(1) process for log productivity.
        par.ygrid = exp(ygrid) # The AR(1) is in logs so exponentiate it to get A.
        par.pmat = pmat # Transition matrix.

        import pandas as pd

        # Inside setup() method, after setting asset grid:

        # Load G_t from CSV
        G_t_data = pd.read_csv('/Users/datho/datalab_export_2025-04-27 17_13_48.csv')
        par.G_t = G_t_data['G_t'].values  # Make sure it's an array indexed by age
        #par.hour = G_t_data['Hours'].values  # Make sure it's an array indexed by age

        # No need for Tauchen anymore
        #par.ygrid = par.G_t.reshape(1, -1)  # Reshape like (1, T)
        #par.pmat = None  # No transition matrix anymore

    
        # Utility function.
        par.util = util
        
        print('beta: ',par.beta)
        print('sigma: ',par.sigma)
        print('amin: ',par.amin)
        print('amax: ',par.amax)
        print('kappa: ',par.kappa)

#%% CRRA Utility Function.
def util(c,hour,sigma,phi,eta):

    # CRRA utility
    if sigma == 1.0:
        u = log(c) # Log utility.
    else:
        u = (c**(1.0-sigma))/(1.0-sigma)-(phi)*(hour)**(1+eta)/(1+eta) # CRRA utility.
    
    return u

#%% Tauchen's Method.
def tauchen(mu,rho,sigma,N,m):
    """
    
    This function discretizes an AR(1) process.
    
            y(t) = mu + rho*y(t-1) + eps(t), eps(t) ~ NID(0,sigma^2)
    
    Input:
        mu    : Intercept of AR(1).
        rho   : Persistence of AR(1).
        sigma : Standard deviation of error term.
        N     : Number of states.
        m     : Parameter such that m time the unconditional std. dev. of the AR(1) is equal to the largest grid point.
        
    Output:
        y    : Grid for the AR(1) process.
        pmat : Transition probability matrix.
        
    """
    
    #%% Construct equally spaced grid.
    
    ar_mean = mu/(1.0-rho) # The mean of a stationary AR(1) process is mu/(1-rho).
    ar_sd = sigma/((1.0-rho**2.0)**(1/2)) # The std. dev of a stationary AR(1) process is sigma/sqrt(1-rho^2)
    
    y1 = ar_mean-(m*ar_sd) # Smallest grid point is the mean of the AR(1) process minus m*std.dev of AR(1) process.
    yn = ar_mean+(m*ar_sd) # Largest grid point is the mean of the AR(1) process plus m*std.dev of AR(1) process.
     
    y,d = linspace(y1,yn,N,endpoint=True,retstep=True) # Equally spaced grid. Include endpoint (endpoint=True) and record stepsize, d (retstep=True).
    
    #%% Compute transition probability matrix from state j (row) to k (column).
    
    ymatk = tile(expand_dims(y,axis=0),(N,1)) # Container for state next period.
    ymatj = mu+rho*ymatk.T # States this period.
    
    # In the following, loc and scale are the mean and std used to standardize the variable. # For example, norm.cdf(x,loc=y,scale=s) is the standard normal CDF evaluated at (x-y)/s.
    pmat = stats.norm.cdf(ymatk,loc=ymatj-(d/2.0),scale=sigma)-stats.norm.cdf(ymatk,loc=ymatj+(d/2.0),scale=sigma) # Transition probabilities to state 2, ..., N-1.
    pmat[:,0] = stats.norm.cdf(y[0],loc=mu+rho*y-(d/2.0),scale=sigma) # Transition probabilities to state 1.
    pmat[:,N-1] = 1.0-stats.norm.cdf(y[N-1],loc=mu+rho*y+(d/2.0),scale=sigma) # Transition probabilities to state N.
    
    #%% Output.
    
    y = expand_dims(y,axis=0) # Convert 0-dimensional array to a row vector.
    
    if count_nonzero(pmat.sum(axis=1)<0.999999) > 0:
        raise Exception("Some columns of transition matrix don't sum to 1.") 

    return y,pmat