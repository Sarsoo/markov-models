import numpy as np

from maths import gaussian

class MarkovModel:
    """Describes a single training iteration including likelihoods and reestimation params"""

    def __init__(self, states: list, observations: list = list(), state_transitions: list = list()):
        self.observations = observations
        self.state_transitions = state_transitions
        # ^ use state number not state index, is padded by entry and exit probs

        self.states = states

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
        """Calculate all likelihoods and both P(O|model)'s"""

        self.populate_forward()
        self.calculate_p_obs_forward()
        self.populate_backward()
        self.calculate_p_obs_backward()
        self.populate_occupation()
        return self
    
    @property
    def observation_likelihood(self):
        """abstraction for getting P(O|model) for future calculations (occupation/transition)"""
        return self.p_obs_forward

    ####################################
    #           Likelihoods
    ####################################
    
    def populate_forward(self):
        """Populate forward likelihoods for all states/times"""
        
        for t, observation in enumerate(self.observations):
            # iterate through observations (time)
            
            for state_index, state in enumerate(self.states):
                # both states at each step

                state_number = state_index + 1 
                # ^ for easier reading (arrays 0-indexed, _number 1-indexed)

                if t == 0: # calcualte initial, 0 = first row = initial
                    self.forward[state_index, t] = self.state_transitions[0, state_number] * gaussian(observation, state.mean, state.std_dev)
                else:
                    # each state for each time has two paths leading to it, 
                    # the same state (this) and the other state (other)

                    other_index = self.get_other_state_index(state_index)
                    other_number = other_index + 1 # for 1 indexing
                    
                    # previous value * prob of changing from previous state to current
                    this_to_this = self.forward[state_index, t - 1] * self.state_transitions[state_number, state_number]
                    other_to_this = self.forward[other_index, t - 1] * self.state_transitions[other_number, state_number]

                    self.forward[state_index, t] = (this_to_this + other_to_this) * gaussian(observation, state.mean, state.std_dev)

        return self.forward

    def calculate_p_obs_forward(self):
        """Calculate, store and return P(O|model) going forwards"""

        sum = 0
        for state_index, final_likelihood in enumerate(self.forward[:, -1]):       
            sum += final_likelihood * self.state_transitions[state_index + 1, -1]
            # get exit prob from state transitions ^

        self.p_obs_forward = sum
        return sum

    def populate_backward(self):
        """Populate backward likelihoods for all states/times"""

        # initialise with exit probabilities
        self.backward[:, -1] = self.state_transitions[1:len(self.states) + 1, -1]

        # below iterator skips first observation 
        # (will be used when finalising P(O|model))
        # iterate backwards through observations (time) [::-1] <- reverses list
        for t, observation in list(enumerate(self.observations[1:]))[::-1]:
            
            # print(t, observation)
            for state_index in range(len(self.states)):

                state_number = state_index + 1 
                # ^ for easier reading (arrays 0-indexed, _number 1-indexed)

                other_index = self.get_other_state_index(state_index)
                other_number = other_index + 1 # for 1 indexing

                # observation for transitions from the same state
                this_state_gaussian = gaussian(observation, self.states[state_index].mean, self.states[state_index].std_dev)
                # observation for transitions from the other state
                other_state_gaussian = gaussian(observation, self.states[other_index].mean, self.states[other_index].std_dev)
                
                # a * b * beta
                this_from_this = self.state_transitions[state_number, state_number] * this_state_gaussian * self.backward[state_index, t + 1]
                other_from_this = self.state_transitions[state_number, other_number] * other_state_gaussian * self.backward[other_index, t + 1]

                self.backward[state_index, t] = this_from_this + other_from_this
        
        return self.backward

    def calculate_p_obs_backward(self):
        """Calculate, store and return P(O|model) going backwards"""

        sum = 0
        for state_index, initial_likelihood in enumerate(self.backward[:, 0]):

            pi = self.state_transitions[0, state_index + 1]
            b = gaussian(self.observations[0], 
                         self.states[state_index].mean, 
                         self.states[state_index].std_dev)
            beta = initial_likelihood

            sum +=  pi * b * beta

        self.p_obs_backward = sum
        return sum

    def populate_occupation(self):
        """Populate occupation likelihoods for all states/times"""

        for t in range(len(self.observations)): 
            # iterate through observations (time)
            
            for state_index in range(len(self.states)):
                
                forward_backward = self.forward[state_index, t] * self.backward[state_index, t]
                self.occupation[state_index, t] = forward_backward / self.observation_likelihood

        return self.occupation

    def transition_likelihood(self, from_index, to_index, t):
        """Get specific transition likelihood given state index either side and the timestep"""
        #from_index = i, from equations in the notes
        #to_index = j, from equations in the notes

        if t == 0:
            print("no transition likelihood for t == 0")

        forward = self.forward[from_index, t - 1]
        transition = self.state_transitions[from_index + 1, to_index + 1]
        emission = gaussian(self.observations[t], 
                            self.states[to_index].mean, 
                            self.states[to_index].std_dev)
        backward = self.backward[to_index, t]

        return (forward * transition * emission * backward) / self.observation_likelihood

    ####################################
    #     Baum-Welch Re-estimations
    ####################################

    def reestimated_state_transitions(self):
        """Re-estimate state transitions using Baum-Welch training (Not on mark scheme)"""

        length = len(self.states)
        new_transitions = np.zeros((length, length))

        # i
        for from_index in range(length):
            # j
            for to_index in range(length):

                # numerator iterates from t = 1 (when 0 indexing, 2 in the notes)
                transition_sum = sum(self.transition_likelihood(from_index, to_index, t) 
                                     for t in range(1, len(self.observations)))
                occupation_sum = sum(self.occupation[from_index, t] 
                                     for t in range(0, len(self.observations)))

                new_transitions[from_index, to_index] = transition_sum / occupation_sum

        return new_transitions

    def reestimated_state_mean(self, state_index):
        """Re-estimate the gaussian mean for a state using occupation likelihoods, baum-welch"""
        
        numerator = 0 # sum over observations( occupation * observation )
        denominator = 0 # sum over observations( occupation )
        for t, observation in enumerate(self.observations): 
            # iterate through observations (time)

            occupation_likelihood = self.occupation[state_index, t]

            numerator += occupation_likelihood * observation
            denominator += occupation_likelihood

        return numerator / denominator

    def reestimated_mean(self):
        """Get all re-estimated gaussian means using occupation likelihoods"""
        return [self.reestimated_state_mean(idx) for idx in range(len(self.states))]


    def reestimated_state_variance(self, state_index):
        """Re-estimate the gaussian variance for a state using occupation likelihoods, baum-welch"""
        
        numerator = 0 # sum over observations( occupation * (observation - mean)^2 )
        denominator = 0 # sum over observations( occupation )
        for t, observation in enumerate(self.observations): 
            # iterate through observations (time)

            occupation_likelihood = self.occupation[state_index, t]

            numerator += occupation_likelihood * pow(observation - self.states[state_index].mean, 2)
            denominator += occupation_likelihood

        return numerator / denominator

    def reestimated_variance(self):
        """Get all re-estimated gaussian variances using occupation likelihoods"""
        return [self.reestimated_state_variance(idx) for idx in range(len(self.states))]
