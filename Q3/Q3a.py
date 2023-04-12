import rasterio
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
from matplotlib import pyplot
from pyopencl.elementwise import ElementwiseKernel

# Import bands as separate images; in /project2/macs30123 on Midway2
band4 = rasterio.open("/project2/macs30123/landsat8/LC08_B4.tif") #red
band5 = rasterio.open("/project2/macs30123/landsat8/LC08_B5.tif") #nir
# Convert nir and red objects to float64 arrays
red = band4.read(1).astype("float64")
nir = band5.read(1).astype("float64")
start = time.time()
ndvi = (nir - red) / (nir + red)
end = time.time()
print(end - start)
# NDVI calculation
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
lin_comb = ElementwiseKernel(ctx,
                             "double *x, double *y, double *ndvi",
                             "ndvi[i] = (x[i] - y[i])/(x[i] + y[i])")
start = time.time()
red_cl = cl_array.to_device(queue, red)
nir_cl = cl_array.to_device(queue, nir)
ndvi = cl.array.empty_like(red_cl)
lin_comb(nir_cl, red_cl, ndvi)
result_np = ndvi.get()
end = time.time()
print(end - start)
pyplot.imsave("Q3a.png", result_np)
