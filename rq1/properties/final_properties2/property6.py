from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 6   # Shirt
b = 5   # Sandal
c = 4   # Coat
Forall(
    x,
    Implies(
	    And(
	       (-3 <= x <= 3)
	    ),
        (abs(N(x)[0, 6] - N(x)[0, 5]) > abs(N(x)[0, 6] - N(x)[0, 4]))
    ),
)