from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 0   # T-shirt/top 
b = 8   # Bag
c = 6   # Shirt
std = Parameter("std", type=float)
Forall(
    x,
    Implies(
	    And(
	       (-std <= x <= std),
		   (argmax(N(x)) == 0),
	    ),
        (abs(N(x)[0, 0] - N(x)[0, 8]) > abs(N(x)[0, 0] - N(x)[0, 6]))
    ),
)