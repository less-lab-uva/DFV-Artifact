from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 3   # Dress
b = 7   # Sneaker
c = 0   # T-shirt
Forall(
    x,
    Implies(
	    And(
	       (0 <= x <= 1)
	    ),
        (abs(N(x)[0, 3] - N(x)[0, 7]) > abs(N(x)[0, 3] - N(x)[0, 0]))
    ),
)