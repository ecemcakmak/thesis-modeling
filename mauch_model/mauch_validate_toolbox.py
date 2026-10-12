import sys
import numpy as np

sys.path.insert(0, "/mauch_data")

from mca_toolbox import calculate_mca
from mauch_ground_truth import CV_r1, parameter_names


parameters = {
    "r1_max": 14.0,
    "km1_s": 0.4,
    "km1_p": 0.2,
    "keq1": 0.7,
    "r2_max": 5.3,
    "km2_s": 0.1,
    "km2_p": 0.05,
    "keq2": 1.6,
    "r3_max": 68.7,
    "km3_s": 0.2,
    "km3_p": 0.01,
    "keq3": 1.2,
    "c_feed": 100.0,
    "D": 0.01,
}


result = calculate_mca(
    "/mauch_data/mauch_synthetic_measurements.csv",
    parameters,
)

toolbox_CV = result["flux_control_R1"]


print("=== MCA TOOLBOX VALIDATION ===")
print()
print(f"{'Parameter':10s} {'Ground truth':>16s} {'Toolbox':>16s} {'Abs. error':>16s}")
print("-" * 62)


errors = []

for name, ground_truth, toolbox in zip(
    parameter_names,
    CV_r1,
    toolbox_CV,
):
    error = abs(ground_truth - toolbox)
    errors.append(error)

    print(
        f"{name:10s} "
        f"{ground_truth:+.12f} "
        f"{toolbox:+.12f} "
        f"{error:.3e}"
    )


max_error = max(errors)

print()
print(f"Maximum absolute error: {max_error:.3e}")

tolerance = 1e-8

if max_error < tolerance:
    print(f"VALIDATION: PASS (tolerance = {tolerance:.1e})")
else:
    print(f"VALIDATION: FAIL (tolerance = {tolerance:.1e})")
