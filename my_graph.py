"""

my_graph.py
-----------
This code plots the value and policy functions.

"""

#%% Imports from Python
from matplotlib.pyplot import figure,plot,xlabel,ylabel,title,show
from numpy import linspace,nanmean,squeeze,zeros

#%% Plot the model functions and simulations.
def track_aging(myClass):
    '''
    
    This function plots the model functions and simulations.
    
    Input:
        myClass : Model class with parameters, grids, utility function, policy functions, and simulations.
        
    '''

    # Model parameters, policy and value functions, and simulations.
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.
    sim = myClass.sim # Simulations.
    
    age = linspace(0,par.T,par.T,endpoint=False,dtype=int)
    
    # Plot savings policy function.

    figure(1)
    plot(age,squeeze(sol.a[0,:,:]))
    xlabel('$Age$')
    ylabel('$a_{t+1}$') 
    title('Savings Policy Function, Lowest $a_t$')

    figure(2)
    plot(age,squeeze(sol.a[par.alen-1,:,:]))
    xlabel('$Age$')
    ylabel('$a_{t+1}$') 
    title('Savings Policy Function, Highest $a_t$')

    # Plot consumption policy function.
    
    figure(3)
    plot(age,squeeze(sol.c[0,:,:]))
    xlabel('$Age$')
    ylabel('$c_{t}$') 
    title('Consumption Policy Function, Lowest $a_t$')
    
    figure(4)
    plot(age,squeeze(sol.c[par.alen-1,:,:]))
    xlabel('$Age$')
    ylabel('$c_{t}$') 
    title('Consumption Policy Function, Highest $a_t$')

    figure(10)
    plot(age,squeeze(sol.h[0,:,:]))
    xlabel('$Age$')
    ylabel('$h_{t}$')
    title('Hours Worked, Lowest $a_t$')

    figure(11)
    plot(age,squeeze(sol.h[par.alen-1,:,:]))
    xlabel('$Age$')
    ylabel('$h_{t}$')
    title('Hours Worked, Highest $a_t$')
    # Plot utility function.

    # Plot value function.
    
    figure(5)
    plot(age,squeeze(sol.v[0,:,:]))
    xlabel('$Age$')
    ylabel('$V_t(k_t)$') 
    title('Value Function, Lowest $a_t$')
    
    figure(6)
    plot(age,squeeze(sol.v[par.alen-1,:,:]))
    xlabel('$Age$')
    ylabel('$V_t(k_t)$') 
    title('Value Function, Highest $a_t$')
    
    # Plot simulated income.

    lcp_c = zeros((par.T,1))
    lcp_a = zeros((par.T,1))
    lcp_u = zeros((par.T,1))

    for i in range(par.T):
        lcp_c[i] = nanmean(sim.csim[sim.tsim==i])
        lcp_a[i] = nanmean(sim.asim[sim.tsim==i])
        lcp_u[i] = nanmean(sim.usim[sim.tsim==i])

    # Plot simulated consumption.
    
    figure(7)
    plot(age,lcp_c)
    xlabel('Age')
    ylabel('$c^{sim}_{t}$') 
    title('Simulated Consumption')
    
    # Plot simulated savings.
    
    figure(8)
    plot(age,lcp_a)
    xlabel('Age')
    ylabel('$a^{sim}_{t+1}$') 
    title('Simulated Savings')

    # Plot simulated utility.
    
    figure(9)
    plot(age,lcp_u)
    xlabel('Age')
    ylabel('$u^{sim}_t$') 
    title('Simulated Utility')
    
    show()
