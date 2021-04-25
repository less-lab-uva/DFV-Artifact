from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 7   # Sneaker
b = 2   # Pullover
c = 9   # Ankle boot
Forall(
    x,
    Implies(
	    And(
	       (0 <= x <= 1)
	    ),
        (abs(N(x)[0, 7] - N(x)[0, 2]) > abs(N(x)[0, 7] - N(x)[0, 9]))
    ),
)