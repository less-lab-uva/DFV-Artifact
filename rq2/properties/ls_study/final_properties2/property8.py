from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 4   # Coat
b = 8   # Bag
c = 2   # Pullover
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std)
	    ),
        (abs(N(x)[0, 4] - N(x)[0, 8]) > abs(N(x)[0, 4] - N(x)[0, 2]))
    ),
)