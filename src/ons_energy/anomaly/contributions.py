"""Variable contributions to T² and SPE for a PCA model on standardized data z.

With loadings P (m×k) and retained eigenvalues λ:
  T²  = zᵀ D z,  D = P Λ⁻¹ Pᵀ
  SPE = ‖C̃ z‖²,  C̃ = I − P Pᵀ

* partial / complete-decomposition contributions (CDC): c_j = z_j (M z)_j, which sum exactly to the
  statistic (M = D for T²). For SPE the classic choice is the squared residual e_j².
* reconstruction-based contributions (RBC, Alcala & Qin 2009): RBC_j = (M z)_j² / M_jj — the
  reduction of the statistic when variable j is reconstructed. Less smearing between correlated
  variables; under a single-sensor fault the faulty variable has the largest RBC.
"""

import numpy as np


def t2_matrix(P: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return P @ np.diag(1.0 / lam) @ P.T


def spe_matrix(P: np.ndarray) -> np.ndarray:
    return np.eye(P.shape[0]) - P @ P.T


def cdc(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Complete-decomposition contributions, rows sum to zᵀMz."""
    return Z * (Z @ M.T)


def spe_partial(Z: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Squared residual per variable, rows sum to SPE."""
    E = Z - (Z @ P) @ P.T
    return E**2


def rbc(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reconstruction-based contributions for the quadratic statistic zᵀMz."""
    diag = np.clip(np.diag(M), 1e-12, None)
    return (Z @ M.T) ** 2 / diag
