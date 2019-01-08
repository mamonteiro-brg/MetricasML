#import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from metricas import *
#import math

# Esqueleto do prof agora adpatar

# ==================  my distances ================
def my_distance_euclidian(x, y):
    if x.size != y.size:
        return (-1)
    nn = Euclidian(x, y)

    return nn

def my_distance_manhattan(x,y):
    if x.size != y.size:
        return (-1)
    nn = Manhattan(x, y)
    return nn

def my_distance_chebyshev(x,y):
    if x.size != y.size:
        return (-1)
    nn = Chebyshev(x, y)
    return nn

def my_distance_canberra(x,y):
    if x.size != y.size:
        return (-1)
    nn = Canberra(x, y)
    return nn

def my_distance_cosine(x,y):
    if x.size != y.size:
        return (-1)
    nn = Cosine(x, y)
    return nn

# ================== end ================


def initialise_representative(data, K):
    return data[0:K, :]


def assigment(data, rep, K, N):
    clusters = np.zeros(N, dtype=np.int)
    for n in range(0, N):
        val_min = 1E20
        for k in range(0, K):
            val = my_distance(data[n, :], rep[k, :])
            if (val < val_min):
                val_min = val
                clusters[n] = k
    return clusters


def centroid_mean(data, clusters, K, N):
    rep = data[0:K, :] * 0
    nb = np.zeros(K, dtype=np.int)
    for n in range(0, N):
        k = clusters[n]
        rep[k, :] = rep[k, :] + data[n, :]
        nb[k] = nb[k] + 1
    for k in range(0, K):
        rep[k, :] = rep[k, :] / nb[k]
    return rep


# def centroid_median(data,cluster):
#
#    return representative

# def medoid(data,cluster):
#
#      return representative
def Error_representative(old_rep, rep, K):
    value = 0
    for k in range(0, K):
        value = value + my_distance(old_rep[k, :], rep[k, :])
    return value


def within_clusters(data, rep, clusters, K, N):
    Sw = np.zeros(K, dtype=np.float)
    nb = np.zeros(K, dtype=np.int)
    for n in range(0, N):
        k = clusters[n]
        Sw[k] = Sw[k] + my_distance(data[n, :], rep[k, :])
        nb[k] = nb[k] + 1
    Sw = np.divide(Sw, nb)
    return Sw


def between_cluster(rep, K):
    Sb = np.ones(K, dtype=np.float) * 1E20
    for k in range(0, K):
        for kk in range(0, K):
            val = my_distance(rep[k, :], rep[kk, :])
            if (k != kk) and val < Sb[k]:
                Sb[k] = val
    return Sb


# ================== MAIN =====================

# read the data file
# my_data = pd.read_csv('../../csv/iris.csv',sep=',');D=np.array(my_data.values[:,1:3],dtype=np.float)
my_data = pd.read_csv("DishonestInternetusersdataset.csv");

print(my_data)
D = np.array(my_data.values[0:3000, 1:3], dtype=np.float)



# set the sizes
N = D.shape[0]  # number of elements/events
d = D.shape[1]  # number of attribute
K = 3  # number of clusters
error = 1;
ncount = 0;
n_MAX = 20
print('number of event:', N, '. number of attribute:', d, ', number of cluster:', K)
# initialise clustering structure
# rep=D[0:K,:]*0  # M=(m1,m2,...,mK)
# old_rep=D[0:K,:]*0
# clusters=np.zeros(N,dtype=np.int)  # C=(c(1),....,c(n),...,c(N)) , c(n) € {1,...,K}.
rep = initialise_representative(D, K)

# test the assigment and centroid
'''
clusters=assigment(D,rep,K,N)
rep=centroid_mean(D,clusters,K,N) 
plt.scatter(D[:,0],D[:,1],c=clusters,s=50,cmap='viridis')
plt.scatter(rep[:, 0],rep[:, 1], c='black', s=200, alpha=0.5);
plt.show()
'''
# main loop
while ((error > 1.E-8) and (ncount < n_MAX)):
    clusters = assigment(D, rep, K, N)
    old_rep = rep
    rep = centroid_mean(D, clusters, K, N)
    error = Error_representative(old_rep, rep, K)
    ncount = ncount + 1
    # plt.scatter(D[:,0],D[:,1],c=clusters,s=50,cmap='viridis')
    # plt.scatter(new_rep[:, 0],new_rep[:, 1], c='black', s=200, alpha=0.5);
    # plt.show()

print('number of iteration:', ncount)
plt.scatter(D[:, 0], D[:, 1], c=clusters, s=50, cmap='viridis')
plt.scatter(rep[:, 0], rep[:, 1], c='black', s=200, alpha=0.5);
plt.show()

# validation
Sw = within_clusters(D, rep, clusters, K, N)
Sb = between_cluster(rep, K)
print("max within clusters error", Sw.max())
print("min between clusters error", Sb.min())
print("Clustering index:", Sw.max() / Sb.min())

# ------- END ------
print('bye')
