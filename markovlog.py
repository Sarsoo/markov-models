from numpy import log as ln

from maths import gaussian
from markov import MarkovModel

# child object to replace normal prob/likeli operations with log prob operations (normal prob for debugging)
class LogMarkovModel(MarkovModel):

    def log_state_transitions(self):
        self.state_transitions = ln(self.state_transitions)