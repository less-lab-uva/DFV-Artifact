from dnnv.properties import *
import numpy as np
N = Network("N")#[:-1]
a = 1   # Trouser
b = 7   # Sneaker
c = 3   # Dress
Forall(
    x,
    Implies(
	    And(
	       (0 <= x <= 1),
		   (argmax(N(x)) == 1),
	    ),
        (abs(N(x)[0, 1] - N(x)[0, 7]) > abs(N(x)[0, 1] - N(x)[0, 3]))
    ),
)