"""

simulate.py
-----------
This code simulates the model.

"""

#%% Imports from Python
from numpy import cumsum,empty,linspace,nan,squeeze,where,zeros
from numpy.random import choice,rand,seed
from numpy.linalg import matrix_power
from types import SimpleNamespace

#%% Simulate the model.
def grow_old_and_die(myClass):
    '''
    
    This function simulates the deterministic life cycle model.
    
    Input:
        myClass : Model class with parameters, grids, utility function, and policy functions.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Simulate the Model')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for simulation.
    setattr(myClass,'sim',SimpleNamespace())
    sim = myClass.sim

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.

    sigma = par.sigma # CRRA.
    util = par.util # Utility function.
    seed_sim = par.seed_sim # Seed for simulation.

    T = par.T # Last period of life.
    tr = par.tr # First year retired.
    
    kappa = par.kappa # Share of income as pension.
    ygrid = par.ygrid # Exogenous income.
    pmat = par.pmat # Transition matrix.
    hour = sol.h # Hours worked.
    phi = par.phi # CRRA.
    eta = par.eta # CRRA.
        
    alen = par.alen # Asset grid size.
    agrid = par.agrid # Assets today (state).
    
    apol = sol.a # Policy function for savings.
    cpol = sol.c # Policy function for consumption.

    TT = par.TT # Time periods.
    NN = par.NN # Population size.
    
    tsim = empty((TT,NN)) * nan # Container for simulated age.
    ysim = empty((TT,NN)) * nan # Container for simulated income.
    csim = empty((TT,NN)) * nan # Container for simulated consumption.
    asim = empty((TT,NN)) * nan # Container for simulated savings.
    hsim = empty((TT,NN)) * nan # Container for simulated hours.
    usim = empty((TT,NN)) * nan # Container for simulated utility.
            
    # Begin simulation.
    
    seed(seed_sim)

    pmat0 = matrix_power(pmat,1000)
    pmat0 = pmat0[0,:] # % Stationary distribution.
    cmat = cumsum(par.pmat,axis=1) # CDF matrix.

    y0_ind = choice(linspace(0,par.ylen,par.ylen,endpoint=False,dtype=int),NN,p=pmat0) # Index for initial income.
    t0_ind = choice(linspace(0,T,T,endpoint=False,dtype=int),NN) # Index for initial age.
    a0_ind = choice(linspace(0,alen,alen,endpoint=False,dtype=int),NN) # Index for initial wealth.
    yr = empty((NN,1)) # Retirement income.

    for i in range(0,NN):
        tsim[0,i] = t0_ind[i] # Simulated age.
        
        if t0_ind[i] < tr: # Simulated income.
            ysim[0,i] = ygrid[0][y0_ind[i]]
        else:
            ysim[0,i] = kappa*ygrid[0][y0_ind[i]]
            
        csim[0,i] = cpol[a0_ind[i],t0_ind[i],y0_ind[i]] # Consumption in period 1 given a0.
        asim[0,i] = apol[a0_ind[i],t0_ind[i],y0_ind[i]] # Savings given a0.
        hsim[0,i] = hour[a0_ind[i],t0_ind[i],y0_ind[i]] # Hours worked in period 1 given a0.
        usim[0,i] = util(csim[0,i],hsim[0,i],sigma,phi,eta) # Utility in period 1 given a0 and age.

        if t0_ind[i] == tr-1: # Retired next period.
            yr[i] = ygrid[0][y0_ind[i]] # Store as pension for next period
        elif t0_ind[i] < tr-1:
            y1_ind = where(rand(1)<=squeeze(cmat[y0_ind[i],:])) # Draw income shock for next period.
            y0_ind[i] = y1_ind[0][0]

    # Simulate endogenous variables.
    
    for j in range(1,TT): # Time loop.
        for i in range(0,NN): # Person loop.
            
            if tsim[j-1,i]+1 <= T-1: # Check if still alive.
                tsim[j,i] = tsim[j-1,i]+1 # Age in period t.
                
                if tsim[j,i] < tr: # Income given age.
                    ysim[j,i] = ygrid[0][y0_ind[i]]
                else:
                    ysim[j,i] = kappa*yr[i]
                
                at_ind = where(asim[j-1,i]==agrid)[0] # Savings choice in the previous period is the state today. Find where the latter is on the grid.
                
                csim[j,i] = cpol[at_ind,int(tsim[j,i]),int(y0_ind[i])] # Consumption in period t.
                asim[j,i] = apol[at_ind,int(tsim[j,i]),int(y0_ind[i])] # Savings for period t+1.
                hsim[j,i] = hour[at_ind,int(tsim[j,i]),int(y0_ind[i])] # Hours for period t+1.
                usim[j,i] = util(csim[j,i],hsim[j,i],sigma,phi,eta) # Utility in period t.
         
                if tsim[j-1,i]+1 == tr-1: # Retire next period
                    yr[i] = ygrid[0][y0_ind[i]] # Store as pension for next period
                elif tsim[j-1,i]+1 < tr-1:
                    y1_ind = where(rand(1)<=squeeze(cmat[y0_ind[i],:])) # Draw income shock for next period.
                    y0_ind[i] = y1_ind[0][0]
                    
    # Simulated model.
    sim.ysim = ysim # Simulated income.
    sim.tsim = tsim # Simulated age.
    sim.csim = csim # Simulated consumption.
    sim.asim = asim # Simulated savings.
    sim.hsim = hsim # Simulated hours.
    sim.usim = usim # Simulated utility.

    # Compute average consumption-to-income ratio (ignore nan and zero denominators)
    import numpy as np
    valid_cy = np.isfinite(csim) & np.isfinite(ysim) & (ysim != 0)
    if np.any(valid_cy):
        avg_c_to_y = np.nanmean(csim[valid_cy] / ysim[valid_cy])
        print(f'Average consumption-to-income ratio: {avg_c_to_y:.4f}')
    else:
        print('No valid data for consumption-to-income ratio.')

    # Compute average consumption-to-saving ratio (ignore nan and zero denominators)
    valid_ca = np.isfinite(csim) & np.isfinite(asim) & (asim != 0)
    if np.any(valid_ca):
        avg_c_to_a = np.nanmean(csim[valid_ca] / asim[valid_ca])
        print(f'Average consumption-to-saving ratio: {avg_c_to_a:.4f}')
    else:
        print('No valid data for consumption-to-saving ratio.')

    # Compute average consumption-to-hour ratio (ignore nan and zero denominators)
    valid_ca = np.isfinite(csim) & np.isfinite(hsim) & (hsim != 0)
    if np.any(valid_ca):
            avg_c_to_h = np.nanmean(csim[valid_ca] / hsim[valid_ca])
            print(f'Average consumption-to-hour ratio: {avg_c_to_h:.4f}')
    else:
            print('No valid data for consumption-to-saving ratio.')
    print('Simulation done.\n')
    print('--------------------------------------------------------------------------------------------------\n')

