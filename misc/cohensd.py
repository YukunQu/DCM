import numpy as np
from numpy import std, mean, sqrt


def cohens_d_onesample(sample_data, mu0):
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    cohens_d = (sample_mean - mu0) / sample_std
    return cohens_d


def cohensd(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s
