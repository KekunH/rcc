import numpy as np
import scipy.stats as sts
import mpi4py
import time
from numba import jit

rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

S = 1000
T = int(4160)
np.random.seed(1203)
eps_mat = sts.norm.rvs(loc = 0, scale = sigma, size = (T,S))
z_mat = np.zeros((T,S))

start = time.perf_counter()
for s_ind in range(S):
  z_tm1 = z_0
  for t_ind in range(T):
    e_t = eps_mat[t_ind, s_ind]
    z_t = rho * z_tm1 + (1 - rho) * mu + e_t
    z_mat[t_ind, s_ind] = z_t
    z_tm1 = z_t
end = time.perf_counter()
print(end - start)

@jit(nopython = True)
def sim_fast(rho, mu, sigma, z_0, S, T, eps_mat, z_mat):
  for s_ind in range(S):
    z_tm1 = z_0
    for t_ind in range(T):
      e_t = eps_mat[t_ind, s_ind]
      z_t = rho * z_tm1 + (1 - rho) * mu + e_t
      z_mat[t_ind, s_ind] = z_t
      z_tm1 = z_t
  return z_mat

start = time.perf_counter()
sim_fast(rho, mu, sigma, z_0, S, T, eps_mat, z_mat)
end = time.perf_counter()
print(end - start)
