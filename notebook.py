# %%
#IMPORTS AND COMMON VARIABLES
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import sqrt

from constants import *
from maths import gaussian
from markov import MarkovModel
from markovlog import LogMarkovModel

x = np.linspace(-4, 8, 120) # x values for figures
x_label = "Observation Space"
y_label = "Probability Density"

# %% [markdown]
# State Probability Functions (1)
# ===================

# %%
state_1_y = [gaussian(i, state1.mean, state1.std_dev) for i in x]
state_2_y = [gaussian(i, state2.mean, state2.std_dev) for i in x]

plt.plot(x, state_1_y, c='r', label="State 1")
plt.plot(x, state_2_y, c='b', label="State 2")

plt.legend()
plt.title("State Probability Density Functions")

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(linestyle="--")

plt.show()

# %% [markdown]
# Output Probability Densities (2)
# ==========

# %%
for obs in observations:
    print(f'{obs} -> State 1: {gaussian(obs, state1.mean, state1.std_dev)}, State 2: {gaussian(obs, state2.mean, state2.std_dev)}')


# %%
state_1_y = [gaussian(i, state1.mean, state1.std_dev) for i in x]
state_2_y = [gaussian(i, state2.mean, state2.std_dev) for i in x]

plt.plot(x, state_1_y, c='r', label="State 1")
plt.plot(x, state_2_y, c='b', label="State 2")

plt.legend()
plt.title("State Probability Density Functions With Observations")

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(linestyle="--")

state1_pd = [gaussian(i, state1.mean, state1.std_dev) for i in observations]
state2_pd = [gaussian(i, state2.mean, state2.std_dev) for i in observations]

#############################################
#             Observation Marks  
#############################################

config = {
    "s": 65,
    "marker": 'x'
}

plt.scatter(observations, state1_pd, color=(0.5, 0, 0), **config)
plt.scatter(observations, state2_pd, color=(0, 0, 0.5), **config)

plt.show()

# %% [markdown]
# # Forward Procedure (3)

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition)
model.populate_forward()

print(model.forward)

forward = model.forward
model.calculate_p_obs_forward()

# %% [markdown]
# # Backward Procedure (4)

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition)
model.populate_backward()

print(model.backward)

backward = model.backward
model.calculate_p_obs_backward()

# %% [markdown]
# # Compare Forward/Backward Final

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition)
model.populate_forward()
model.populate_backward()

print("forward:", model.calculate_p_obs_forward())
print("backward:", model.calculate_p_obs_backward())

print("diff: ", model.p_obs_forward - model.p_obs_backward)

# %% [markdown]
# # Occupation Likelihoods (5)

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition).populate()

occupation = model.occupation
print(model.occupation)

# %% [markdown]
# # Re-estimate Mean & Variance (6)

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition).populate()

print("mean: ", [state1.mean, state2.mean])
print("variance: ", [state1.variance, state2.variance])
print()

print("mean: ", model.reestimated_mean())
print("variance: ", model.reestimated_variance())

# %% [markdown]
# New PDFs (7)
# ===================

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition).populate()

new_mean = model.reestimated_mean()
new_var = model.reestimated_variance()
new_std_dev = [sqrt(x) for x in new_var]

state_1_y = [gaussian(i, new_mean[0], new_std_dev[0]) for i in x]
state_2_y = [gaussian(i, new_mean[1], new_std_dev[1]) for i in x]

plt.plot(x, state_1_y, c='r', label="State 1")
plt.plot(x, state_2_y, c='b', label="State 2")

plt.legend()
plt.title("Re-estimated Probability Density Functions")

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(linestyle="--")

plt.show()

# %% [markdown]
# # Compare PDFs (7)

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition).populate()

new_mean = model.reestimated_mean()
new_var = model.reestimated_variance()
new_std_dev = [sqrt(x) for x in new_var]

#######################################
#              Original
#######################################
state_1_y = [gaussian(i, state1.mean, state1.std_dev) for i in x]
state_2_y = [gaussian(i, state2.mean, state2.std_dev) for i in x]
plt.plot(x, state_1_y, '--', c='r', label="State 1", linewidth=1.0)
plt.plot(x, state_2_y, '--', c='b', label="State 2", linewidth=1.0)

#######################################
#            Re-Estimated
#######################################
state_1_new_y = [gaussian(i, new_mean[0], new_std_dev[0]) for i in x]
state_2_new_y = [gaussian(i, new_mean[1], new_std_dev[1]) for i in x]
plt.plot(x, state_1_new_y, c='r', label="New State 1")
plt.plot(x, state_2_new_y, c='b', label="New State 2")

plt.legend()
plt.title("Re-estimated Probability Density Functions")

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(linestyle="--")

plt.show()

# %% [markdown]
# # Multiple Iterations

# %%
iterations = 5

mean = [state1.mean, state2.mean]
var = [state1.variance, state2.variance]

plt.plot(x, [gaussian(i, mean[0], sqrt(var[0])) for i in x], '--', c='r', linewidth=1.0)
plt.plot(x, [gaussian(i, mean[1], sqrt(var[1])) for i in x], '--', c='b', linewidth=1.0)

for i in range(iterations):
    model = MarkovModel(states=[State(mean[0], var[0], state1.entry, state1.exit), State(mean[1], var[1], state2.entry, state2.exit)], 
                        observations=observations, 
                        state_transitions=state_transition)
    model.populate()

    mean = model.reestimated_mean()
    var = model.reestimated_variance()

    print(f"mean ({i}): ", mean)
    print(f"var ({i}): ", var)
    print()

    state_1_y = [gaussian(i, mean[0], sqrt(var[0])) for i in x]
    state_2_y = [gaussian(i, mean[1], sqrt(var[1])) for i in x]

    style = '--'
    linewidth = 1.0
    if i == iterations - 1:
        style = '-'
        linewidth = 2.0

    plt.plot(x, state_1_y, style, c='r', linewidth=linewidth)
    plt.plot(x, state_2_y, style, c='b', linewidth=linewidth)

plt.title("Probability Density Function Iterations")

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(linestyle="--")

plt.show()

# %% [markdown]
# # Baum-Welch State Transition Re-estimations

# %%
model = MarkovModel(states=[state1, state2], observations=observations, state_transitions=state_transition).populate()

print(a_matrix)
model.reestimated_state_transitions()


# %%



