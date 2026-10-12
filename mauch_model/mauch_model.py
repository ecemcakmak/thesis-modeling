import numpy as np


def reversible_mm(cs, cp, rmax, km_s, km_p, keq):
    return rmax * (cs - cp / keq) / (
        km_s * (1 + cp / km_p) + cs
    )


# Parameters from Mauch et al. (1997)
r1_max = 14.0
r2_max = 5.3
r3_max = 68.7

km1_s = 0.4
km1_p = 0.2
keq1 = 0.7

km2_s = 0.1
km2_p = 0.05
keq2 = 1.6

km3_s = 0.2
km3_p = 0.01
keq3 = 1.2

c_feed = 100.0
D = 0.01

# Reported initial steady state
c1_0 = 0.2
c2_0 = 0.1
cP_0 = 0.1
cQ_0 = 0.1


# Calculate reaction rates at the reported steady state
r1_0 = reversible_mm(c1_0, c2_0, r1_max, km1_s, km1_p, keq1)
r2_0 = reversible_mm(c2_0, cP_0, r2_max, km2_s, km2_p, keq2)
r3_0 = reversible_mm(c2_0, cQ_0, r3_max, km3_s, km3_p, keq3)

if __name__ == "__main__":
    print("r1 =", r1_0)
    print("r2 =", r2_0)
    print("r3 =", r3_0)
