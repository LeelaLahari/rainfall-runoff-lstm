"""
Nash-Sutcliffe Efficiency (NSE) — the standard hydrology evaluation metric.
Using this instead of plain MSE is a meaningful domain-knowledge signal.
"""
import numpy as np


def nse(obs, sim):
    obs, sim = np.asarray(obs), np.asarray(sim)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - obs.mean()) ** 2)
    return 1 - numerator / (denominator + 1e-8)


def rmse(obs, sim):
    return np.sqrt(np.mean((np.asarray(obs) - np.asarray(sim)) ** 2))