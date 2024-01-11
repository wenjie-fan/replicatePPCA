import copy
import numpy as np


def nipals(data, max_iter=500, tol=1e-6, n_components=2):
    Y = copy.deepcopy(data)
    n, m = Y.shape
    Y_mean = np.nanmean(Y, axis=0)
    Y = Y - Y_mean
    # Replace NaN values with 0
    Y[np.isnan(Y)] = 0
    
    T = np.zeros((n, n_components))
    P = np.zeros((m, n_components))
    
    for h in range(n_components):
        # Initialize t with the column of Y with maximum variance
        t = Y[:, np.nanvar(Y, axis=0).argmax()].reshape((-1, 1))
        for _ in range(max_iter):
            p = np.dot(Y.T, t) / np.dot(t.T, t)
            p /= np.linalg.norm(p)
            t_old = t
            t = np.dot(Y, p) / np.dot(p.T, p)
            if np.linalg.norm(t - t_old) < tol:
                break
                
        T[:, h] = t.ravel()
        P[:, h] = p.ravel()
        # Deflation
        Y -= np.outer(t, p.T)
    
    # Return Transformation T, PCs P, Reconstructions Y_recon
    Y_recon = np.dot(T, P.T) + Y_mean
    Y_impute = np.where(np.isnan(data), Y_recon, data)
    return T, P, Y_recon, Y_impute