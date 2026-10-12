from mauch_model import (
    reversible_mm,
    r1_max, r2_max, r3_max,
    km1_s, km1_p, keq1,
    km2_s, km2_p, keq2,
    km3_s, km3_p, keq3,
    c_feed, D,
    c1_0, c2_0, cP_0, cQ_0
)


# Calculate reaction rates at the reported initial state
r1_0 = reversible_mm(
    c1_0, c2_0,
    r1_max, km1_s, km1_p, keq1
)

r2_0 = reversible_mm(
    c2_0, cP_0,
    r2_max, km2_s, km2_p, keq2
)

r3_0 = reversible_mm(
    c2_0, cQ_0,
    r3_max, km3_s, km3_p, keq3
)


# Calculate time derivatives
dc1_dt = D * (c_feed - c1_0) - r1_0
dc2_dt = r1_0 - r2_0 - r3_0


print("=== Mauch 1997 validation ===")
print()
print("Reaction rates:")
print(f"r1 = {r1_0:.6f}")
print(f"r2 = {r2_0:.6f}")
print(f"r3 = {r3_0:.6f}")

print()
print("Steady-state residuals:")
print(f"dc1/dt = {dc1_dt:.6f}")
print(f"dc2/dt = {dc2_dt:.6f}")
