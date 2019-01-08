import math
import numpy as np

def Euclidian(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    nn = math.sqrt((z**2).sum())
    return nn


def Manhattan(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    nn = np.fabs(z).sum()
    return nn


def Chebyshev(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    nn = np.amax(np.fabs(z))
    return nn


def Canberra(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    numer = np.fabs(z)
    denom = np.fabs(x) + np.fabs(y)
    r = np.true_divide(numer, denom)
    nn = r.sum()
    return nn


def Cosine(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    numer = np.dot(x, y)
    denom = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    nn = 1 - numer / denom
    return nn
