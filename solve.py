"""

solve.py
--------
This code solves the model.

"""

#%% Imports from Python
from numpy import argmax,expand_dims,exp,log,max,inf,squeeze,tile,unravel_index,zeros,seterr
from types import SimpleNamespace
import time
seterr(divide='ignore')
seterr(invalid='ignore')

#%% Solve the model using Backward Induction.
def experience_life(myClass):
    '''
    
    This function solves the stochastic life cycle model.
    
    Input:
        myClass : Model class with parameters, grids, and utility function.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Solving the Model by Backward Induction')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for optimal policy funtions.
    setattr(myClass,'sol',SimpleNamespace())
    sol = myClass.sol

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    
    T = par.T # Last period of life.
    tr = par.tr # First year of retirement.
    
    beta = par.beta # Discount factor.
    sigma = par.sigma # CRRA.
    phi = par.phi # CRRA.
    eta = par.eta # CRRA.
    G_t = par.G_t # Growth rate of income.

    hlen = par.hlen # Grid size for h.
    hours = par.hours # Hours worked.
    
    alen = par.alen # Grid size for a.
    agrid = par.agrid # Grid for a (state and choice).

    ylen = par.ylen # Grid size for y.
    ygrid = par.ygrid # Grid for y.
    pmat = par.pmat # Transition matrix for y.

    r = par.r # Real interest rate.
    kappa = par.kappa # Share of income as pension.

    util = par.util # Utility function.

    amat = tile(expand_dims(agrid,axis=1),(1,hlen)) # a' for each value of a and h
    ymat = tile(expand_dims(ygrid,axis=0),(alen,1)) # A for each value of k.
    hmat = tile(expand_dims(hours,axis=0),(alen,1)) # h for each value of a and h

    # Containers.
    v1 = zeros((alen,T,ylen)) # Container for V.
    a1 = zeros((alen,T,ylen)) # Container for a'.
    c1 = zeros((alen,T,ylen)) # Container for c.
    h1 = zeros((alen,T,ylen)) # Container for c.

    t0 = time.time()

    for age in reversed(range(T)): # Iterate on the Bellman Equation until convergence.
        for i in range(ylen): # Loop over the y-states.
            yt = G_t[age]*ygrid[0][i]*hmat  # yt is (alen, hlen), hmat is (alen, hlen)
            if age == T-1:
                ev = zeros((alen, hlen))
            else:
                ev_next = squeeze(v1[:,age+1,:]) @ pmat[i,:].T  # shape (alen,)
                ev = tile(ev_next.reshape(-1,1), (1, hlen))     # shape (alen, hlen)
            for p in range(alen): 
                # Consumption.
                ct = agrid[p] + yt - (amat / (1.0 + r))  # ct is (alen, hlen)
                ct[ct<0.0] = 0.0
                
                # Solve the maximization problem.
                vall = util(ct, hmat, sigma, phi, eta) + beta * ev
                vall[ct <= 0.0] = -inf
                ind_max = unravel_index(argmax(vall), vall.shape)
                v1[p, age, i] = vall[ind_max]
                a1[p, age, i] = agrid[ind_max[0]]
                c1[p, age, i] = ct[ind_max]
                h1[p, age, i] = hmat[ind_max]
                
        # Print counter.
        if age%5 == 0:
            print('Age: ',age,'.\n')

    t1 = time.time()
    print('Elapsed time is ',t1-t0,' seconds.')

    # Macro variables, value, and policy functions.
    sol.c = c1 # Consumption policy function.
    sol.a = a1 # Saving policy function.
    sol.v = v1 # Value function.
    sol.h = h1 # Hours function.

    print(sol.h)