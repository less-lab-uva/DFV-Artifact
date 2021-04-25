from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 6   # Shirt
b = 9   # Ankle boot
c = 2   # Pullover
Forall(
    x,
    Implies(
	    And(
	       (-3 <= x <= 3),
		   (argmax(N(x)) == 6),
	    ),
        (abs(N(x)[0, 6] - N(x)[0, 9]) > abs(N(x)[0, 6] - N(x)[0, 2]))
    ),
)