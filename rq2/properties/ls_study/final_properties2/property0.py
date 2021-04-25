from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 7   # Sneaker
b = 6   # Shirt
c = 5   # Sandal
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std)
	    ),
        (abs(N(x)[0, 7] - N(x)[0, 6]) > abs(N(x)[0, 7] - N(x)[0, 5]))
    ),
)