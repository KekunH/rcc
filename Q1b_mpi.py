import numpy as np
import scipy.stats as sts
from mpi4py import MPI
import time
from numba import jit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = time.time()
rho = 0.5
mu = 3.0
sigma = 1.0
T = int(4160)
S = int(1000/size)

@jit(nopython = True)
def sim_fast(rho, mu, sigma, S, T, eps_mat):
  z_0 = mu
  z_mat = np.zeros((T,S))
  for s_ind in range(S):
    z_tm1 = z_0
    for t_ind in range(T):
      e_t = eps_mat[t_ind, s_ind]
      z_t = rho * z_tm1 + (1 - rho) * mu + e_t
      z_mat[t_ind, s_ind] = z_t
      z_tm1 = z_t
  return z_mat

np.random.seed(rank)
eps_mat = sts.norm.rvs(loc = 0, scale = sigma, size = (T,S))
results = sim_fast(rho, mu, sigma, S, T, eps_mat)

results_all = None
if rank == 0:
  results_all = np.empty([S * size, 4160], dtype='float')
comm.Gather(sendbuf=results, recvbuf=results_all, root=0)

if rank == 0:
  time_elapsed = time.time() - start
  print(time_elapsed, size)
