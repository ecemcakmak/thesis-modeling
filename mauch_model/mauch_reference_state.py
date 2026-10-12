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


def steady_state_equations(x):
    c1, c2 = x

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

    dc1_dt = D * (c_feed - c1) - r1
    dc2_dt = r1 - r2 - r3

    return np.array([dc1_dt, dc2_dt])


def numerical_jacobian(func, x, step=1e-7):
    """Calculate the Jacobian using central finite differences."""

    x = np.array(x, dtype=float)
    n = len(x)

    J = np.zeros((n, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += step
        x_minus[j] -= step

        J[:, j] = (
            func(x_plus) - func(x_minus)
        ) / (2 * step)

    return J


def find_steady_state(initial_guess, tolerance=1e-12, max_iterations=50):
    """Find the steady state using Newton's method."""

    x = np.array(initial_guess, dtype=float)

    for iteration in range(max_iterations):

        residual = steady_state_equations(x)

        if np.max(np.abs(residual)) < tolerance:
            return x, iteration + 1

        J = numerical_jacobian(steady_state_equations, x)

        correction = np.linalg.solve(J, -residual)

        x = x + correction

    raise RuntimeError(
        "Steady-state calculation did not converge."
    )


# Mauch's reported steady-state concentrations
initial_guess = [0.2, 0.1]

# Find mathematically consistent steady state
solution, iterations = find_steady_state(initial_guess)

c1_ss, c2_ss = solution


# Reaction rates at the consistent steady state
r1_ss = reversible_mm(
    c1_ss, c2_ss,
    r1_max, km1_s, km1_p, keq1
)

r2_ss = reversible_mm(
    c2_ss, cP_0,
    r2_max, km2_s, km2_p, keq2
)

r3_ss = reversible_mm(
    c2_ss, cQ_0,
    r3_max, km3_s, km3_p, keq3
)


# Final steady-state residuals
residuals = steady_state_equations(solution)


if __name__ == "__main__":

    print("=== Consistent Mauch reference steady state ===")
    print()

    print("Concentrations:")
    print(f"c1 = {c1_ss:.9f}")
    print(f"c2 = {c2_ss:.9f}")
    print(f"cP = {cP_0:.9f}")
    print(f"cQ = {cQ_0:.9f}")

    print()
    print("Reaction rates:")
    print(f"r1 = {r1_ss:.9f}")
    print(f"r2 = {r2_ss:.9f}")
    print(f"r3 = {r3_ss:.9f}")

    print()
    print("Steady-state residuals:")
    print(f"dc1/dt = {residuals[0]:.3e}")
    print(f"dc2/dt = {residuals[1]:.3e}")

    print()
    print(f"Newton iterations = {iterations}")
