from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 5   # Sandal
b = 1   # Trouser
c = 7   # Sneaker
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std)
	    ),
        (abs(N(x)[0, 5] - N(x)[0, 1]) > abs(N(x)[0, 5] - N(x)[0, 7]))
    ),
)