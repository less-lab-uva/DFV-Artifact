from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 0   # T-shirt/top 
b = 8   # Bag
c = 6   # Shirt
Forall(
    x,
    Implies(
	    And(
	       (-3 <= x <= 3),
		   (argmax(N(x)) == 0),
	    ),
        (abs(N(x)[0, 0] - N(x)[0, 8]) > abs(N(x)[0, 0] - N(x)[0, 6]))
    ),
)