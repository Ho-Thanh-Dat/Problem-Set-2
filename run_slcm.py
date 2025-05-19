"""

run_slcm.py
-----------
This code solves the stochastic life cycle model using value function iteration.

"""

#%% Import from folder
from model import household
from solve import experience_life
from simulate import grow_old_and_die
from my_graph import track_aging
import numpy as np
import matplotlib.pyplot as plt

#%% Stochastic Growth Model.
some_dude = household()

# Set the parameters, state space, and utility function.
some_dude.setup(beta = 0.94,sigma=2.00) # You can set the parameters here or use the defaults.

# Solve the model.
experience_life(some_dude) # Obtain the policy functions for consumption and savings.

# Simulate the model.
grow_old_and_die(some_dude) # Simulate forward in time.

# Graphs.
track_aging(some_dude) # Plot policy functions and simulations.


