import csv

from mauch_model import (
    reversible_mm,
    r1_max, r2_max, r3_max,
    km1_s, km1_p, keq1,
    km2_s, km2_p, keq2,
    km3_s, km3_p, keq3,
    cP_0, cQ_0
)

from mauch_reference_state import find_steady_state


# ------------------------------------------------------------
# Reference steady state
# ------------------------------------------------------------

reference_state, iterations = find_steady_state(
    [0.2, 0.1]
)

c1, c2 = reference_state


# ------------------------------------------------------------
# Calculate reaction rates
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Synthetic measurement dataset
# ------------------------------------------------------------

data = {
    "c1": c1,
    "c2": c2,
    "cP": cP_0,
    "cQ": cQ_0,
    "r1": r1,
    "r2": r2,
    "r3": r3
}


# ------------------------------------------------------------
# Save as CSV
# ------------------------------------------------------------

filename = "mauch_synthetic_measurements.csv"

with open(filename, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "c1",
        "c2",
        "cP",
        "cQ",
        "r1",
        "r2",
        "r3"
    ])

    writer.writerow([
        f"{c1:.12f}",
        f"{c2:.12f}",
        f"{cP_0:.12f}",
        f"{cQ_0:.12f}",
        f"{r1:.12f}",
        f"{r2:.12f}",
        f"{r3:.12f}"
    ])


# ------------------------------------------------------------
# Print dataset
# ------------------------------------------------------------

print("=== Mauch synthetic measurement dataset ===")
print()

for variable, value in data.items():
    print(f"{variable:>3} = {value:.12f}")

print()
print(f"Saved to: {filename}")
