# Load LightGBM's libomp before anything imports torch: importing torch first makes multi-threaded
# LightGBM segfault on macOS. Tests that *run* torch ops after LightGBM was used go to a subprocess.
import lightgbm  # noqa: F401
