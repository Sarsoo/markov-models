from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)  # implements constructor among other boilerplate
class State:
    mean: float
    variance: float
    
    entry: float  # pi
    exit: float  # eta


state1 = State(1, 1.44, 0.44, 0.02)
state2 = State(4, 0.49, 0.56, 0.03)
        
observations = [3.8, 4.2, 3.4, -0.4, 1.9, 3.0, 1.6, 1.9, 5.0]

a_matrix = np.array([[0.92, 0.06],
                     [0.04, 0.93]])

state_transition = np.array([[0, 0.44, 0.56, 0], 
                             [0, 0.92, 0.06, 0.02], 
                             [0, 0.04, 0.93, 0.03], 
                             [0, 0, 0, 0]])
