import numpy as np

from mauch_model import (
    reversible_mm,
    r1_max, r2_max, r3_max,
    km1_s, km1_p, keq1,
    km2_s, km2_p, keq2,
    km3_s, km3_p, keq3,
    c_feed, D,
    cP_0, cQ_0
)

from mauch_reference_state import (
    find_steady_state
)


# ============================================================
# 1. Reference steady state
# ============================================================

initial_guess = [0.2, 0.1]

c_ss, iterations = find_steady_state(initial_guess)

c1, c2 = c_ss


# ============================================================
# 2. Reaction rates at steady state
# ============================================================

r1 = reversible_mm(
    c1, c2,
    r1_max, km1_s, km1_p, keq1
)

r2 = reversible_mm(
    c2, cP_0,
    r2_max, km2_s, km2_p, keq2
)

r3 = reversible_mm(
    c2, cQ_0,
    r3_max, km3_s, km3_p, keq3
)


# ============================================================
# 3. Analytical reaction derivatives
# ============================================================

def reaction_derivatives(cs, cp, rmax, km_s, km_p, keq):
    """
    Analytical derivatives of reversible Michaelis-Menten rate.

    Returns:
        v
        dv/dcs
        dv/dcp
        dv/drmax
        dv/dkm_s
        dv/dkm_p
    """

    numerator = cs - cp / keq

    denominator = (
        km_s * (1 + cp / km_p)
        + cs
    )

    v = rmax * numerator / denominator

    dv_dcs = (
        rmax * (denominator - numerator)
        / denominator**2
    )

    dv_dcp = (
        rmax
        * (
            (-1 / keq) * denominator
            - numerator * (km_s / km_p)
        )
        / denominator**2
    )

    dv_drmax = numerator / denominator

    dv_dkm_s = (
        -rmax
        * numerator
        * (1 + cp / km_p)
        / denominator**2
    )

    dv_dkm_p = (
        rmax
        * numerator
        * km_s
        * cp
        / (km_p**2 * denominator**2)
    )

    return (
        v,
        dv_dcs,
        dv_dcp,
        dv_drmax,
        dv_dkm_s,
        dv_dkm_p
    )


(
    _,
    r1_c1,
    r1_c2,
    r1_rmax,
    r1_kms,
    r1_kmp
) = reaction_derivatives(
    c1, c2,
    r1_max, km1_s, km1_p, keq1
)


(
    _,
    r2_c2,
    r2_cp,
    r2_rmax,
    r2_kms,
    r2_kmp
) = reaction_derivatives(
    c2, cP_0,
    r2_max, km2_s, km2_p, keq2
)


(
    _,
    r3_c2,
    r3_cp,
    r3_rmax,
    r3_kms,
    r3_kmp
) = reaction_derivatives(
    c2, cQ_0,
    r3_max, km3_s, km3_p, keq3
)


# ============================================================
# 4. Stoichiometric matrix
# ============================================================

N = np.array([
    [-1.0,  0.0,  0.0],
    [ 1.0, -1.0, -1.0]
])


# ============================================================
# 5. Reaction-rate Jacobian wrt metabolites
# ============================================================

dv_dc = np.array([
    [r1_c1, r1_c2],
    [0.0,   r2_c2],
    [0.0,   r3_c2]
])


# Jacobian of the full dynamic system
#
# f1 = D(c_feed - c1) - r1
# f2 = r1 - r2 - r3

J = N @ dv_dc

# Contribution from the feed/dilution term
J[0, 0] -= D


# ============================================================
# 6. Parameter vector
# ============================================================

parameter_names = [
    "r1_max",
    "KM1_S",
    "KM1_P",
    "r2_max",
    "KM2_S",
    "KM2_P",
    "r3_max",
    "KM3_S",
    "KM3_P"
]

parameters = np.array([
    r1_max,
    km1_s,
    km1_p,
    r2_max,
    km2_s,
    km2_p,
    r3_max,
    km3_s,
    km3_p
])


# ============================================================
# 7. Direct reaction-rate derivatives wrt parameters
# ============================================================

dv_dp = np.zeros((3, 9))

# Reaction I
dv_dp[0, 0] = r1_rmax
dv_dp[0, 1] = r1_kms
dv_dp[0, 2] = r1_kmp

# Reaction II
dv_dp[1, 3] = r2_rmax
dv_dp[1, 4] = r2_kms
dv_dp[1, 5] = r2_kmp

# Reaction III
dv_dp[2, 6] = r3_rmax
dv_dp[2, 7] = r3_kms
dv_dp[2, 8] = r3_kmp


# Derivative of system equations wrt parameters
Phi = N @ dv_dp


# ============================================================
# 8. Normalized concentration control coefficients
# ============================================================

X_inv = np.diag([
    1.0 / c1,
    1.0 / c2
])

P = np.diag(parameters)

CM = (
    -X_inv
    @ np.linalg.solve(J, Phi)
    @ P
)


# ============================================================
# 9. Flux control coefficients for r1
# ============================================================

elasticity_r1 = np.array([
    c1 / r1 * r1_c1,
    c2 / r1 * r1_c2
])

parameter_elasticity_r1 = (
    parameters / r1
) * dv_dp[0, :]


CV_r1 = (
    elasticity_r1 @ CM
    + parameter_elasticity_r1
)


# ============================================================
# 10. Print results
# ============================================================

print("=== Mauch 1997 analytical MCA ground truth ===")
print()

print("Reference steady state:")
print(f"c1 = {c1:.9f}")
print(f"c2 = {c2:.9f}")
print(f"r1 = {r1:.9f}")
print(f"r2 = {r2:.9f}")
print(f"r3 = {r3:.9f}")

print()
print("Jacobian:")
print(J)

print()
print("Flux control coefficients for r1:")
print()

for name, value in zip(parameter_names, CV_r1):
    print(f"{name:10s} : {value:+.9f}")
