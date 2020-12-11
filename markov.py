from dataclasses import dataclass, field
from typing import List
import numpy as np

from maths import gaussian

@dataclass
class Likelihood:
    forward: float  # forward likelihood
    backward: float  # backward likelihood

@dataclass
class TimeStep:
    states: List[Likelihood] = field(default_factory=list)

@dataclass
class Transition:
    pass

class MarkovModel:

    def __init__(self, states: list, observations: list = list(), state_transitions: list = list()):
        self.observations = observations
        self.state_transitions = state_transitions

        self.states = states # number of states
        # self.timesteps = list()

        self.forward = np.zeros((len(states), len(observations)))
        self.backward = np.zeros((len(states), len(observations)))
    
    def populate_forward(self):
        for t, observation in enumerate(self.observations): # iterate through observations (time)
            for state_number, state in enumerate(self.states):

                if t == 0: # calcualte initial
                    self.forward[state_number, t] = self.state_transitions[0, state_number + 1] * gaussian(observation, state.mean, state.std_dev)
                else:
                    self.forward[state_number, t] = gaussian(observation, state.mean, state.std_dev)
