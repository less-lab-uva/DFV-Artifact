from dnnv.properties import *
import numpy as np

VAE = Network("VAE")
DNN = Network("DNN")
N = DNN[2:].compose(VAE)
N_prob_coll = N[:-2, 1]
N_steer_angle = N[:-1, 0]

logit = lambda x: np.log(x / (1 - x))
P_coll_min = logit(0.6)
P_coll_max = logit(0.7)

steer_max = 60 * np.pi / 180

Forall(
    x,
    Implies(
        And(-3 <= x <= 3, P_coll_min < N_prob_coll(x) <= P_coll_max),
        -steer_max <= N_steer_angle(x) <= steer_max,
    ),
)
