from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 5   # Sandal
b = 2   # Pullover
c = 9   # Ankle boot
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std),
		   (argmax(N(x)) == 5),
	    ),
        (abs(N(x)[0, 5] - N(x)[0, 2]) > abs(N(x)[0, 5] - N(x)[0, 9]))
    ),
)