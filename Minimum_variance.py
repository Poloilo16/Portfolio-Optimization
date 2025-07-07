import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf

# Helper function: convert correlation matrix and std to covariance matrix
# If you have a different implementation, replace this stub
def corr2cov(corr, std):
    # cov = diag(std) * corr * diag(std)
    return np.outer(std, std) * corr

def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    np.fill_diagonal(block, 1)
    corr = block_diag(*([block] * nBlocks))
    return corr

def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0

# Example usage for block matrix and true matrix generation
nBlocks, bSize, bCorr = 10, 50, 0.5
np.random.seed(0)
mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)


def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov

#---------------------------------------------------
def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    np.fill_diagonal(block, 1)
    corr = block_diag(*([block] * nBlocks))
    return corr

#---------------------------------------------------
def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0

# Example usage for block matrix and true matrix generation
nBlocks, bSize, bCorr = 10, 50, 0.5
np.random.seed(0)
mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)


def optPort(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

# Stub for simCovMu (replace with your actual implementation)
def simCovMu(mu, cov, nObs, shrink=False):
    # Simulate a sample mean and covariance matrix
    X = np.random.multivariate_normal(mu.flatten(), cov, size=nObs)
    mu1 = X.mean(axis=0).reshape(-1, 1)
    cov1 = np.cov(X, rowvar=False)
    return mu1, cov1

# Stub for deNoiseCov (replace with your actual implementation)
def deNoiseCov(cov0, q, bWidth):
    # For now, just return the input covariance
    return cov0

# Portfolio simulation parameters
nObs, nTrials, bWidth, shrink, minVarPortf = 1000, 1000, 0.01, False, True

def run_portfolio_simulation(mu0, cov0, nObs, nTrials, bWidth, shrink, minVarPortf):
    w1 = pd.DataFrame(columns=range(cov0.shape[0]), index=range(nTrials), dtype=float)
    w1_d = w1.copy(deep=True)
    np.random.seed(0)
    for i in range(nTrials):
        mu1, cov1 = simCovMu(mu0, cov0, nObs, shrink=shrink)
        if minVarPortf:
            mu1 = None
        cov1_d = deNoiseCov(cov1, nObs * 1.0 / cov1.shape[1], bWidth)
        w1.loc[i] = optPort(cov1, mu1).flatten()
        w1_d.loc[i] = optPort(cov1_d, mu1).flatten()
    w0 = optPort(cov0, None if minVarPortf else mu0)
    w0 = np.repeat(w0.T, w1.shape[0], axis=0)
    rmsd = np.mean((w1.values - w0) ** 2) ** 0.5  # RMSE
    rmsd_d = np.mean((w1_d.values - w0) ** 2) ** 0.5  # RMSE
    return rmsd, rmsd_d

# Parameters
nObs, nTrials, bWidth, minVarPortf = 1000, 1000, 0.01, True

# Not shrunk
rmsd_ns, rmsd_d_ns = run_portfolio_simulation(mu0, cov0, nObs, nTrials, bWidth, shrink=False, minVarPortf=minVarPortf)
print("Not shrunk: RMSE (raw):", rmsd_ns, "RMSE (denoised):", rmsd_d_ns)

# Shrunk
rmsd_s, rmsd_d_s = run_portfolio_simulation(mu0, cov0, nObs, nTrials, bWidth, shrink=True, minVarPortf=minVarPortf)
print("Shrunk: RMSE (raw):", rmsd_s, "RMSE (denoised):", rmsd_d_s)