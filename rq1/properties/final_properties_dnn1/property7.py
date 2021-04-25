from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 5   # Sandal
b = 2   # Pullover
c = 9   # Ankle boot
Forall(
    x,
    Implies(
	    And(
	       (0 <= x <= 1),
		   (argmax(N(x)) == 5),
	    ),
        (abs(N(x)[0, 5] - N(x)[0, 2]) > abs(N(x)[0, 5] - N(x)[0, 9]))
    ),
)