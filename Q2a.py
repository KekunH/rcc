from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import mpi4py
import time
from numba import jit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rho = 0.5
mu = 3.0
sigma = 1.0
S = int(1000)
T = int(4160)


@jit(nopython = True)
def sim_fast_rhos(rhos, mu, sigma, S, T, eps_mat):
  z_mat = np.zeros((T,S))
  z_0 = mu - 3*sigma
  longest_period = 0
  for rho in rhos:
    lst = []
    for s_ind in range(S):
      z_tm1 = z_0
      for t_ind in range(T):
        e_t = eps_mat[t_ind, s_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t
        z_tm1 = z_t
        if z_tm1 <= 0:
          lst.append(t_ind)
          break
    if np.array(lst).mean() > longest_period:
      longest_period = np.array(lst).mean()
      best_rho = rho        
  return (best_rho, longest_period)

start = time.time()
data = None
eps_mat = None
if rank == 0:
  np.random.seed(rank)
  eps_mat = sts.norm.rvs(loc = 0, scale = sigma, size = (T,S))
  data = np.linspace(-0.95, 0.95, 200)
subdata = np.empty(20)
sub_eps = np.empty((416,S))
comm.Scatter(data, subdata, root = 0)
comm.Scatter(eps_mat, sub_eps, root = 0)

result = np.array(sim_fast_rhos(subdata, mu, sigma, S, 416, sub_eps))
all_result = None
if rank == 0:
  all_result = np.empty((2,10))
comm.Gather(result, all_result, root = 0)
if rank == 0:
  long_t = 0
  for val in all_result:
    rho = val[0]
    t = val[1]
    if t > long_t:
      long_t = t
      best_rho = rho
  print(best_rho)
  end = time.time()
  print(end - start)

