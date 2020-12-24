from dataclasses import dataclass, field
from typing import List
import numpy as np
from numpy import log as ln

from maths import gaussian

class MarkovModel:

    def __init__(self, states: list, observations: list = list(), state_transitions: list = list()):
        self.observations = observations
        self.state_transitions = state_transitions # use state number not state index, is padded by entry and exit probs

        self.states = states # number of states
        # self.timesteps = list()

        self.forward = np.zeros((len(states), len(observations)))
        self.p_obs_forward = 0

        self.backward = np.zeros((len(states), len(observations)))
        self.p_obs_backward = 0

        self.occupation = np.zeros((len(states), len(observations)))

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

    def populate(self):
        self.populate_forward()
        self.calculate_p_obs_forward()
        self.populate_backward()
        self.calculate_p_obs_backward()
        self.populate_occupation()
    
    def populate_forward(self):
        for t, observation in enumerate(self.observations): # iterate through observations (time)
            for state_index, state in enumerate(self.states):

                state_number = state_index + 1 # for easier reading (arrays 0-indexed, numbers start at 1)

                if t == 0: # calcualte initial, 0 = first row = initial
                    self.forward[state_index, t] = self.state_transitions[0, state_number] * gaussian(observation, state.mean, state.std_dev)
                else:
                    # each state for each time has two paths leading to it, the same state (this) and the other state (other)

                    other_index = self.get_other_state_index(state_index)
                    other_number = other_index + 1 # for 1 indexing
                    
                    # previous value * prob of changing from previous state to current
                    this_to_this = self.forward[state_index, t - 1] * self.state_transitions[state_number, state_number]
                    other_to_this = self.forward[other_index, t - 1] * self.state_transitions[other_number, state_number]

                    self.forward[state_index, t] = (this_to_this + other_to_this) * gaussian(observation, state.mean, state.std_dev)

    @property
    def observation_likelihood(self):
        """abstraction for getting p(obs|model) for future calculations (occupation/transition)"""
        return self.p_obs_forward

    def calculate_p_obs_forward(self):

        sum = 0
        for state_index, final_likelihood in enumerate(self.forward[:, -1]):            
            sum += final_likelihood * self.state_transitions[state_index + 1, -1] # get exit prob from state transitions

        self.p_obs_forward = sum
        return sum

    def populate_backward(self):

        # initialise from exit probabilities
        self.backward[:, -1] = self.state_transitions[1:len(self.states) + 1, -1]

        for t, observation in list(enumerate(self.observations[1:]))[::-1]: # iterate backwards through observations (time)
            # print(t, observation)
            for state_index, state in enumerate(self.states):

                state_number = state_index + 1 # for easier reading (arrays 0-indexed, numbers start at 1)

                other_index = self.get_other_state_index(state_index)
                other_number = other_index + 1 # for 1 indexing

                # observation for transitions from the same state
                this_state_gaussian = gaussian(observation, self.states[state_index].mean, self.states[state_index].std_dev)
                # observation for transitions from the other state
                other_state_gaussian = gaussian(observation, self.states[other_index].mean, self.states[other_index].std_dev)
                
                # beta * a * b
                this_from_this = self.backward[state_index, t + 1] * self.state_transitions[state_number, state_number] * this_state_gaussian
                other_from_this = self.backward[other_index, t + 1] * self.state_transitions[other_number, state_number] * other_state_gaussian

                self.backward[state_index, t] = (this_from_this + other_from_this)

    def calculate_p_obs_backward(self):

        sum = 0
        for state_index, initial_likelihood in enumerate(self.backward[:, 0]):
            # pi * b * beta
            sum += self.state_transitions[0, state_index + 1] * gaussian(self.observations[0], self.states[state_index].mean, self.states[state_index].std_dev) * initial_likelihood

        self.p_obs_backward = sum
        return sum

    def populate_occupation(self):
        for t, observation in enumerate(self.observations): # iterate through observations (time)
            for state_index, state in enumerate(self.states):
                
                forward_backward = self.forward[state_index, t] * self.backward[state_index, t]
                self.occupation[state_index, t] = forward_backward / self.observation_likelihood

    def transition_likelihood(self, from_index, to_index, t):
        if t == 0:
            print("no transition likelihood for t == 0")

        forward = self.forward[from_index, t - 1]
        transition = self.state_transitions[from_index + 1, to_index + 1]
        emission = gaussian(self.observations[t], self.states[to_index].mean, self.states[to_index].std_dev)
        backward = self.backward[to_index, t]

        return (forward * transition * emission * backward) / self.observation_likelihood

    def baum_welch_state_transitions(self):

        new_transitions = np.zeros((len(self.states), len(self.states)))

        # i
        for from_index, from_state in enumerate(self.states):
            # j
            for to_index, to_state in enumerate(self.states):
                
                transition_sum = 0
                for t in range(1, len(self.observations)):
                    transition_sum += self.transition_likelihood(from_index, to_index, t)

                occupation_sum = 0
                for t in range(0, len(self.observations)):
                    occupation_sum = self.occupation[to_index, t]

                new_transitions[from_index, to_index] = transition_sum / occupation_sum

        return new_transitions



# child object to replace normal prob/likeli operations with log prob operations (normal prob for debugging)
class LogMarkovModel(MarkovModel):

    def log_state_transitions(self):
        self.state_transitions = ln(self.state_transitions)
