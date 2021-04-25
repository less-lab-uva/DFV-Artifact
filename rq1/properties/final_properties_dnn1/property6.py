from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 2   # Pullover
b = 1   # Trouser
c = 4   # Coat
Forall(
    x,
    Implies(
	    And(
	       (0 <= x <= 1),
		   (argmax(N(x)) == 2),
	    ),
        (abs(N(x)[0, 2] - N(x)[0, 1]) > abs(N(x)[0, 2] - N(x)[0, 4]))
    ),
)