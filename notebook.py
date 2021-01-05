# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#IMPORTS AND COMMON VARIABLES
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
from math import sqrt

from constants import *
from maths import gaussian
from markov import MarkovModel
from markovlog import LogMarkovModel

fig_dpi = 200
fig_export = False

x = np.linspace(-4, 8, 300) # x values for figures
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

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
fig.set_tight_layout(True)
if fig_export:
    savefig("report/res/pdfs.png")
plt.show()

# %% [markdown]
# Output Probability Densities (2)
# ==========

# %%
for obs in observations:
    print(f'{obs} -> State 1: {gaussian(obs, state1.mean, state1.std_dev)},', 
                   f'State 2: {gaussian(obs, state2.mean, state2.std_dev)}')


# %%
state_1_y = [gaussian(i, state1.mean, state1.std_dev) for i in x]
state_2_y = [gaussian(i, state2.mean, state2.std_dev) for i in x]

plt.plot(x, state_1_y, c='r', label="State 1")
plt.plot(x, state_2_y, c='b', label="State 2")

plt.legend()
plt.title("State Probability Density Functions With Observations")

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(linestyle="--", axis='y')

state1_pd = [gaussian(i, state1.mean, state1.std_dev) for i in observations]
state2_pd = [gaussian(i, state2.mean, state2.std_dev) for i in observations]

#############################################
#             Observation Marks  
#############################################

config = {
    "s": 65,
    "marker": 'x'
}

[plt.axvline(x=i, ls='--', lw=1.0, c=(0,0,0), alpha=0.4) for i in observations]
plt.scatter(observations, state1_pd, color=(0.5, 0, 0), **config)
plt.scatter(observations, state2_pd, color=(0, 0, 0.5), **config)

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
fig.set_tight_layout(True)
if fig_export:
    savefig("report/res/pdfs-w-obs.png")
plt.show()

# %% [markdown]
# # Forward Procedure (3)

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition)
model.populate_forward()

print(model.forward)

forward = model.forward
model.calculate_p_obs_forward()


# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

state_x = np.arange(1, 10)

from numpy import log as ln

plt.plot(state_x, [ln(i) for i in model.forward[0, :]], c='r', label="State 1")
plt.plot(state_x, [ln(i) for i in model.forward[1, :]], c='b', label="State 2")

plt.ylim(top=0)

plt.legend()
plt.title("Forward Log-Likelihoods Over Time")

plt.xlabel("Observation (t)")
plt.ylabel("Log Likelihood")
plt.grid(linestyle="--")

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
fig.set_tight_layout(True)
if fig_export:
    savefig("report/res/forward-logline.png")
plt.show()

# %% [markdown]
# # Backward Procedure (4)

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition)
model.populate_backward()

print(model.backward)

backward = model.backward
model.calculate_p_obs_backward()


# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

state_x = np.arange(1, 10)

from numpy import log as ln

plt.plot(state_x, [ln(i) for i in model.backward[0, :]], c='r', label="State 1")
plt.plot(state_x, [ln(i) for i in model.backward[1, :]], c='b', label="State 2")

plt.ylim(top=0)

plt.legend()
plt.title("Backward Log-Likelihoods Over Time")

plt.xlabel("Observation (t)")
plt.ylabel("Log Likelihood")
plt.grid(linestyle="--")

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
fig.set_tight_layout(True)
if fig_export:
    savefig("report/res/backward-logline.png")
plt.show()

# %% [markdown]
# # Compare Forward/Backward Final

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition)
model.populate_forward()
model.populate_backward()

print("forward:", model.calculate_p_obs_forward())
print("backward:", model.calculate_p_obs_backward())

print("diff: ", model.p_obs_forward - model.p_obs_backward)

# %% [markdown]
# # Occupation Likelihoods (5)

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

occupation = model.occupation
print(model.occupation)


# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

fig = plt.figure(figsize=(6,6), dpi=fig_dpi, tight_layout=True)
ax = fig.add_subplot(1, 1, 1, projection="3d", xmargin=0, ymargin=0)

y_width = 0.3

X = np.arange(1, 10) - 0.5
Y = np.arange(1, 3) - 0.5*y_width
X, Y = np.meshgrid(X, Y)
Z = np.zeros(model.forward.size)

dx = np.ones(model.forward.size)
dy = y_width * np.ones(model.forward.size)

colours = [*[(1.0, 0.1, 0.1) for i in range(9)], *[(0.2, 0.2, 1.0) for i in range(9)]]
ax.bar3d(X.flatten(), Y.flatten(), Z, 
         dx, dy, model.occupation.flatten(), 
         color=colours, shade=True)

ax.set_yticks([1, 2])
ax.set_zlim(top=1.0)

ax.set_title("Occupation Likelihoods Over Time")
ax.set_xlabel("Observation")
ax.set_ylabel("State")
ax.set_zlabel("Occupation Likelihood")
ax.view_init(35, -72)
if fig_export:
    savefig("report/res/occupation-bars.png")
fig.show()


# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

state_x = np.arange(1, 10)

plt.plot(state_x, model.occupation[0, :], c='r', label="State 1")
plt.plot(state_x, model.occupation[1, :], c='b', label="State 2")

plt.legend()
plt.title("Occupation Likelihoods Over Time")

plt.xlabel("Observation (t)")
plt.ylabel("Occupation Likelihood")
plt.grid(linestyle="--")

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
fig.set_tight_layout(True)
if fig_export:
    savefig("report/res/occupation-line.png")
plt.show()

# %% [markdown]
# # Re-estimate Mean & Variance (6)

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

print("mean: ", [state1.mean, state2.mean])
print("variance: ", [state1.variance, state2.variance])
print()

print("mean: ", model.reestimated_mean())
print("variance: ", model.reestimated_variance())

# %% [markdown]
# New PDFs (7)
# ===================

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

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

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
plt.show()

# %% [markdown]
# # Compare PDFs (7)

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

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

fig = matplotlib.pyplot.gcf()
fig.set_dpi(fig_dpi)
fig.set_tight_layout(True)
if fig_export:
    savefig("report/res/re-est-pdfs.png")
plt.show()

# %% [markdown]
# # Multiple Iterations

# %%
iterations = 50

fig = plt.figure(dpi=fig_dpi, tight_layout=True)
ax = fig.add_subplot(1, 1, 1, xmargin=0, ymargin=0)

iter_mean = [state1.mean, state2.mean]
iter_var = [state1.variance, state2.variance]
iter_state_transitions = state_transition

ax.plot(x, [gaussian(i, iter_mean[0], sqrt(iter_var[0])) for i in x], '--', c='r', linewidth=1.0)
ax.plot(x, [gaussian(i, iter_mean[1], sqrt(iter_var[1])) for i in x], '--', c='b', linewidth=1.0)

label1=None
label2=None

for i in range(iterations):
    iter_model = MarkovModel(states=[State(iter_mean[0], iter_var[0], state1.entry, state1.exit), 
                                     State(iter_mean[1], iter_var[1], state2.entry, state2.exit)],
                             observations=observations,
                             state_transitions=iter_state_transitions).populate()

    # NEW PARAMETERS
    iter_mean = iter_model.reestimated_mean()
    iter_var = iter_model.reestimated_variance()
    iter_state_transitions[1:3, 1:3] = iter_model.reestimated_state_transitions()

    print(f"mean ({i}): ", iter_mean)
    print(f"var ({i}): ", iter_var)
    print(iter_model.reestimated_state_transitions())
    print()

    state_1_y = [gaussian(i, iter_mean[0], sqrt(iter_var[0])) for i in x]
    state_2_y = [gaussian(i, iter_mean[1], sqrt(iter_var[1])) for i in x]

    style = '--'
    linewidth = 1.0
    if i == iterations - 1:
        style = '-'
        linewidth = 2.0
        label1='State 1'
        label2='State 2'

    ax.plot(x, state_1_y, style, c='r', label=label1, linewidth=linewidth)
    ax.plot(x, state_2_y, style, c='b', label=label2, linewidth=linewidth)

ax.set_title("Probability Density Function Iterations")

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.grid(linestyle="--")
ax.legend()

if fig_export:
    savefig("report/res/iterated-pdfs.png")
fig.show()

# %% [markdown]
# # Baum-Welch State Transition Re-estimations

# %%
model = MarkovModel(states=[state1, state2], 
                    observations=observations, 
                    state_transitions=state_transition).populate()

print(a_matrix)
print(model.reestimated_state_transitions())
model.reestimated_state_transitions()


# %%



