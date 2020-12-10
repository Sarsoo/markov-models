from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    mean: float
    variance: float
    
    entry: float  # pi
    exit: float  # eta


state1 = State(1, 1.44, 0.44, 0.02)
state2 = State(4, 0.49, 0.56, 0.03)
        
observations = [3.8, 4.2, 3.4, -0.4, 1.9, 3.0, 1.6, 1.9, 5.0]
