from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 6   # Shirt
b = 9   # Ankle boot
c = 2   # Pullover
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std),
		   (argmax(N(x)) == 6),
	    ),
        (abs(N(x)[0, 6] - N(x)[0, 9]) > abs(N(x)[0, 6] - N(x)[0, 2]))
    ),
)