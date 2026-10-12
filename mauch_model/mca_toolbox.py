import numpy as np
import csv


PARAMETERS = [
    "r1_max",
    "km1_s",
    "km1_p",
    "r2_max",
    "km2_s",
    "km2_p",
    "r3_max",
    "km3_s",
    "km3_p",
]


def reaction_rates(c1, c2, cP, cQ, p):
    """Mauch (1997) reversible Michaelis-Menten reaction rates."""
    r1 = p["r1_max"] * (c1 - c2 / p["keq1"]) / (
        p["km1_s"] * (1.0 + c2 / p["km1_p"]) + c1
    )

    r2 = p["r2_max"] * (c2 - cP / p["keq2"]) / (
        p["km2_s"] * (1.0 + cP / p["km2_p"]) + c2
    )

    r3 = p["r3_max"] * (c2 - cQ / p["keq3"]) / (
        p["km3_s"] * (1.0 + cQ / p["km3_p"]) + c2
    )

    return np.array([r1, r2, r3], dtype=float)


def perturbation_derivative(func, x, relative_step=1e-6):
    """Central finite-difference derivative."""
    h = relative_step * max(abs(x), 1.0)
    return (func(x + h) - func(x - h)) / (2.0 * h)


def calculate_mca(measurement_csv, parameters):
    """
    Calculate MCA coefficients from steady-state measurements.

    The toolbox uses:
        C^c_p = -J^{-1} Phi
    followed by flux-control propagation for R1.
    """

    with open(measurement_csv, newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    required_columns = ["c1", "c2", "cP", "cQ", "r1", "r2", "r3"]
    missing = [c for c in required_columns if c not in data[0]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(data) != 1:
        raise ValueError("This Step 1 toolbox expects exactly one reference steady state.")

    row = data[0]

    c1 = float(row["c1"])
    c2 = float(row["c2"])
    cP = float(row["cP"])
    cQ = float(row["cQ"])

    measured_fluxes = np.array(
        [float(row["r1"]), float(row["r2"]), float(row["r3"])]
    )

    concentrations = np.array([c1, c2])

    def rates_from_concentrations(c):
        return reaction_rates(c[0], c[1], cP, cQ, parameters)

    # ------------------------------------------------------------
    # Concentration elasticities: dr_i / dc_j
    # ------------------------------------------------------------
    epsilon_c = np.zeros((3, 2))

    for j in range(2):
        def f(x):
            c = concentrations.copy()
            c[j] = x
            return rates_from_concentrations(c)

        epsilon_c[:, j] = perturbation_derivative(f, concentrations[j])

    # ------------------------------------------------------------
    # Parameter elasticities: dr_i / dp
    # ------------------------------------------------------------
    epsilon_p = np.zeros((3, len(PARAMETERS)))

    for j, name in enumerate(PARAMETERS):
        p0 = float(parameters[name])
        h = 1e-6 * max(abs(p0), 1.0)

        p_plus = dict(parameters)
        p_minus = dict(parameters)

        p_plus[name] = p0 + h
        p_minus[name] = p0 - h

        r_plus = reaction_rates(c1, c2, cP, cQ, p_plus)
        r_minus = reaction_rates(c1, c2, cP, cQ, p_minus)

        epsilon_p[:, j] = (r_plus - r_minus) / (2.0 * h)

    # ------------------------------------------------------------
    # Jacobian of the two dynamic state equations:
    #
    # dc1/dt = D(c_feed-c1) - r1
    # dc2/dt = r1-r2-r3
    #
    # ------------------------------------------------------------
    J = np.array(
        [
            [-parameters["D"] - epsilon_c[0, 0], -epsilon_c[0, 1]],
            [
                epsilon_c[0, 0],
                epsilon_c[0, 1] - epsilon_c[1, 1] - epsilon_c[2, 1],
            ],
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Phi = partial F / partial p
    # ------------------------------------------------------------
    Phi = np.zeros((2, len(PARAMETERS)))

    for j in range(len(PARAMETERS)):
        Phi[0, j] = -epsilon_p[0, j]
        Phi[1, j] = (
            epsilon_p[0, j]
            - epsilon_p[1, j]
            - epsilon_p[2, j]
        )

    # Concentration control coefficients
    Cc = -np.linalg.solve(J, Phi)

    # ------------------------------------------------------------
    # Flux control coefficients for R1
    #
    # dJ_R1/dp = partial(r1)/partial(p)
    #            + dr1/dc * dc/dp
    #
    # normalized:
    # C^J_p = (p / r1) * dJ_R1/dp
    # ------------------------------------------------------------
    r1 = float(measured_fluxes[0])

    flux_control = np.zeros(len(PARAMETERS))

    for j, name in enumerate(PARAMETERS):
        parameter_value = float(parameters[name])

        total_derivative = (
            epsilon_p[0, j]
            + np.dot(epsilon_c[0, :], Cc[:, j])
        )

        flux_control[j] = (
            parameter_value / r1
        ) * total_derivative

    return {
        "concentrations": concentrations,
        "measured_fluxes": measured_fluxes,
        "epsilon_c": epsilon_c,
        "epsilon_p": epsilon_p,
        "J": J,
        "Phi": Phi,
        "Cc": Cc,
        "flux_control_R1": flux_control,
    }
