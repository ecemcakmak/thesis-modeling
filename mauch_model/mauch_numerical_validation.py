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

from mauch_reference_state import find_steady_state


# ------------------------------------------------------------
# Steady-state equations
# ------------------------------------------------------------

def steady_state_equations(x, parameters):
    c1, c2 = x

    r1 = reversible_mm(
        c1, c2,
        parameters["r1_max"],
        parameters["KM1_S"],
        parameters["KM1_P"],
        keq1
    )

    r2 = reversible_mm(
        c2, cP_0,
        parameters["r2_max"],
        parameters["KM2_S"],
        parameters["KM2_P"],
        keq2
    )

    r3 = reversible_mm(
        c2, cQ_0,
        parameters["r3_max"],
        parameters["KM3_S"],
        parameters["KM3_P"],
        keq3
    )

    return np.array([
        D * (c_feed - c1) - r1,
        r1 - r2 - r3
    ])


# ------------------------------------------------------------
# Numerical Jacobian
# ------------------------------------------------------------

def numerical_jacobian(func, x, parameters, step=1e-7):

    x = np.array(x, dtype=float)
    n = len(x)

    J = np.zeros((n, n))

    for j in range(n):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += step
        x_minus[j] -= step

        J[:, j] = (
            func(x_plus, parameters)
            - func(x_minus, parameters)
        ) / (2 * step)

    return J


# ------------------------------------------------------------
# Solve steady state
# ------------------------------------------------------------

def solve_steady_state(parameters, initial_guess):

    x = np.array(initial_guess, dtype=float)

    for _ in range(50):

        residual = steady_state_equations(x, parameters)

        if np.max(np.abs(residual)) < 1e-12:
            return x

        J = numerical_jacobian(
            steady_state_equations,
            x,
            parameters
        )

        correction = np.linalg.solve(J, -residual)

        x = x + correction

    raise RuntimeError("Steady-state solution did not converge.")


# ------------------------------------------------------------
# Base parameters
# ------------------------------------------------------------

parameters = {
    "r1_max": r1_max,
    "KM1_S": km1_s,
    "KM1_P": km1_p,

    "r2_max": r2_max,
    "KM2_S": km2_s,
    "KM2_P": km2_p,

    "r3_max": r3_max,
    "KM3_S": km3_s,
    "KM3_P": km3_p
}


parameter_names = list(parameters.keys())


# ------------------------------------------------------------
# Reference steady state
# ------------------------------------------------------------

reference_state = solve_steady_state(
    parameters,
    [0.2, 0.1]
)

c1_ref, c2_ref = reference_state


def calculate_r1(state, params):

    c1, c2 = state

    return reversible_mm(
        c1,
        c2,
        params["r1_max"],
        params["KM1_S"],
        params["KM1_P"],
        keq1
    )


# ------------------------------------------------------------
# Numerical perturbation
# ------------------------------------------------------------

perturbation = 1e-5

numerical_control = []


for parameter_name in parameter_names:

    p0 = parameters[parameter_name]

    parameters_plus = parameters.copy()
    parameters_minus = parameters.copy()

    parameters_plus[parameter_name] = (
        p0 * np.exp(perturbation)
    )

    parameters_minus[parameter_name] = (
        p0 * np.exp(-perturbation)
    )

    state_plus = solve_steady_state(
        parameters_plus,
        reference_state
    )

    state_minus = solve_steady_state(
        parameters_minus,
        reference_state
    )

    flux_plus = calculate_r1(
        state_plus,
        parameters_plus
    )

    flux_minus = calculate_r1(
        state_minus,
        parameters_minus
    )

    # Definition:
    # C = d ln(r1) / d ln(parameter)

    coefficient = (
        np.log(flux_plus)
        - np.log(flux_minus)
    ) / (2 * perturbation)

    numerical_control.append(coefficient)


# ------------------------------------------------------------
# Analytical values from mauch_ground_truth.py
# ------------------------------------------------------------

analytical_control = np.array([
    0.000614792,
    -0.000461118,
    0.000153787,
    0.000202385,
    -0.000151759,
    0.000101173,
    0.000203340,
    -0.000194492,
    0.000176811
])


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

print("=== Numerical validation of analytical MCA ===")
print()

print(
    f"{'Parameter':<10}"
    f"{'Analytical':>15}"
    f"{'Numerical':>15}"
    f"{'Difference':>15}"
)

print("-" * 55)

for name, analytical, numerical in zip(
    parameter_names,
    analytical_control,
    numerical_control
):

    difference = numerical - analytical

    print(
        f"{name:<10}"
        f"{analytical:>15.9e}"
        f"{numerical:>15.9e}"
        f"{difference:>15.3e}"
    )
