from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 4   # Coat
b = 1   # Trouser
c = 6   # Shirt
Forall(
    x,
    Implies(
	    And(
	       (-3 <= x <= 3)
	    ),
        (abs(N(x)[0, 4] - N(x)[0, 1]) > abs(N(x)[0, 4] - N(x)[0, 6]))
    ),
)