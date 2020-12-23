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
        self.state_transitions = state_transitions # use state number not state index, is padded by entry and exit probs

        self.states = states # number of states
        # self.timesteps = list()

        self.forward = np.zeros((len(states), len(observations)))
        self.backward = np.zeros((len(states), len(observations)))

    def get_other_state_index(self, state_in):
        """For when state changes, get other index for retrieving state transitions (FOR 0 INDEXING)"""
        if state_in == 0:
            return 1
        elif state_in == 1:
            return 0
        else:
            print(f"invalid state index provided, ({state_in})")

    def get_other_state_number(self, state_in):
        """For when state changes, get other number for retrieving state transitions (FOR 1 INDEXING)"""
        return self.get_other_state_index(state_in - 1) + 1
    
    def populate_forward(self):
        for t, observation in enumerate(self.observations): # iterate through observations (time)
            for state_index, state in enumerate(self.states):

                state_number = state_index + 1 # for easier reading (arrays 0-indexed, numbers start at 1)

                if t == 0: # calcualte initial
                    self.forward[state_index, t] = self.state_transitions[0, state_number] * gaussian(observation, state.mean, state.std_dev)
                else:
                    # each state for each time has two paths leading to

                    other_index = self.get_other_state_index(state_index)
                    other_number = other_index + 1 # for 1 indexing
                    
                    #                      previous value             prob of changing from previous state to current
                    this_to_this = self.forward[state_index, t - 1] * self.state_transitions[state_number, state_number]
                    other_to_this = self.forward[other_index, t - 1] * self.state_transitions[other_number, state_number]

                    self.forward[state_index, t] = (this_to_this + other_to_this) * gaussian(observation, state.mean, state.std_dev)

    @property
    def p_observations_forward(self):

        sum = 0
        for state_index, final_likelihood in enumerate(self.forward[:, -1]):            
            sum += final_likelihood * self.state_transitions[state_index + 1, -1] # get exit prob from state transitions

        return sum

    #TODO finish
    def populate_backward(self):

        # initialise from exit probabilities
        self.backward[:, -1] = self.state_transitions[1:len(self.states) + 1, -1]

        for t, observation in list(enumerate(self.observations[1:]))[::-1]: # iterate backwards through observations (time)
            print(t, observation)
            for state_index, state in enumerate(self.states):

                state_number = state_index + 1 # for easier reading (arrays 0-indexed, numbers start at 1)

                other_index = self.get_other_state_index(state_index)
                other_number = other_index + 1 # for 1 indexing
                
                #                      previous value             prob of changing from previous state to current
                this_to_this = self.backward[state_index, t + 1] * self.state_transitions[state_number, state_number]
                other_to_this = self.backward[other_index, t + 1] * self.state_transitions[other_number, state_number]

                self.backward[state_index, t] = (this_to_this + other_to_this) * gaussian(observation, state.mean, state.std_dev)

    #TODO finish
    @property
    def p_observations_backward(self):

        sum = 0
        for state_index, initial_likelihood in enumerate(self.backward[:, 0]):            
            sum += self.state_transitions[0, state_index + 1] * gaussian(self.observations[0], self.states[state_index].mean, self.states[state_index].std_dev) * initial_likelihood

        return sum
