from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 5   # Sandal
b = 8   # Bag
c = 7   # Sneaker
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std),
		   (argmax(N(x)) == 5),
	    ),
        (abs(N(x)[0, 5] - N(x)[0, 8]) > abs(N(x)[0, 5] - N(x)[0, 7]))
    ),
)