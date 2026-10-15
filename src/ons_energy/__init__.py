"""Load forecasting and multivariate anomaly detection on ONS open data.

macOS caveat: torch and LightGBM link different libomp builds; using both in ONE process segfaults
(exit 139) in either order. Each notebook (kernel) therefore uses a single family (LightGBM *or*
torch), and the torch-based tests run in a subprocess.
"""

__version__ = "0.1.0"
