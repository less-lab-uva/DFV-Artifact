from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 9   # Ankle boot
b = 2   # T-shirt/top
c = 7   # Sneaker
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std),
		   (argmax(N(x)) == 9),
	    ),
        (abs(N(x)[0, 9] - N(x)[0, 2]) > abs(N(x)[0, 9] - N(x)[0, 7]))
    ),
)