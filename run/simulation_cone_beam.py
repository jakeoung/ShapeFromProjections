import toml
from ctdr.utils.util_sino import read_proj_info

# Make a data folder 
# 2testA: the number of materials is 2 and A means the simple type of homogeneous material
ddata = "../data/2testA/"
# !mkdir $ddata
ftoml = f"{ddata}/proj_geom.toml"
nangles = 30

mode = "parallel"

if mode == "cone": # see cone_geometry.jpg
    with open(ftoml, "w") as target:
        target.write('''type = "cone"
    ProjectionAngles = [
        0.0, 0.10471975511965978, 0.20943951023931956, 0.3141592653589793, 0.4188790204786391, 0.5235987755982988, 0.6283185307179586, 0.7330382858376184, 0.8377580409572782, 0.9424777960769379, 1.0471975511965976, 1.1519173063162575, 1.2566370614359172, 1.361356816555577, 1.4660765716752369, 1.5707963267948966, 1.6755160819145565, 1.7802358370342162, 1.8849555921538759, 1.9896753472735358, 2.0943951023931953, 2.199114857512855, 2.303834612632515, 2.4085543677521746, 2.5132741228718345, 2.6179938779914944, 2.722713633111154, 2.827433388230814, 2.9321531433504737, 3.036872898470133]
    DetectorRowCount = 300
    DetectorColCount = 300
    DetectorSpacingX = 0.01041666666666667
    DetectorSpacingY = 0.01041666666666667
    DistanceOriginSource = 10.0
    DistanceOriginDetector = 4.0''')
    
elif mode == "parallel":
    with open(ftoml, "w") as target:
        target.write('''type = "parallel3d"
    ProjectionAngles = [
        0.0, 0.10471975511965978, 0.20943951023931956, 0.3141592653589793, 0.4188790204786391, 0.5235987755982988, 0.6283185307179586, 0.7330382858376184, 0.8377580409572782, 0.9424777960769379, 1.0471975511965976, 1.1519173063162575, 1.2566370614359172, 1.361356816555577, 1.4660765716752369, 1.5707963267948966, 1.6755160819145565, 1.7802358370342162, 1.8849555921538759, 1.9896753472735358, 2.0943951023931953, 2.199114857512855, 2.303834612632515, 2.4085543677521746, 2.5132741228718345, 2.6179938779914944, 2.722713633111154, 2.827433388230814, 2.9321531433504737, 3.036872898470133]
    DetectorRowCount = 300
    DetectorColCount = 300
    DetectorSpacingX = 0.01041666666666667
    DetectorSpacingY = 0.01041666666666667''')
    
else:
    pass

proj_geom = read_proj_info(ftoml, convert_vec=True)

# Download an example watertight obj file
# Download an example watertight obj file
# !wget https://groups.csail.mit.edu/graphics/classes/6.837/F03/models/teddy.obj
# !mv teddy.obj ../data/2testA/init.obj

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import h5py
import time

import ctdr
from ctdr.model.vanilla import Model
from ctdr.utils import util_mesh
from ctdr import optimize
import subprocess
import toml
 
nmaterials = 2
finit_obj = ddata+'/init.obj'
mus = [0.0, 1.0]  # attentuation coefficient

model = Model(finit_obj, proj_geom, nmaterials,
              mus, 0, 0.0, 0.0).cuda()

# (optional) Since the obj file is not normalized to -1 ~ 1, I normalize
# Please check the projection geometry above

def normalize(a):
    a = (a - a.min()) / (a.max() - a.min()) * 1.8 - 0.9
    return a

model.vertices[:,0] = normalize(model.vertices[:,0])
model.vertices[:,1] = normalize(model.vertices[:,1])
model.vertices[:,2] = normalize(model.vertices[:,2])
# model.vertices = model.vertices.type()
# model.mus = model.mus.type(torch.float64)
# print(model.mus.dtype)

# Generate projections
out = model(torch.arange(nangles), 0.0, backprop=False)
proj = out[0].detach().cpu().numpy() # [nangles x height x width]

# Show the projection image
# For non-manifold obj file, there can be some artifacts
plt.imshow(proj[0,:,:]); plt.colorbar(); plt.show()
plt.savefig(f"../result/p0_{mode}.png")
plt.plot()
plt.imshow(proj[4,:,:]); plt.colorbar(); plt.show()
plt.savefig(f"../result/p4_{mode}.png")
plt.imshow(proj[11,:,:]); plt.colorbar(); plt.show()
plt.savefig(f"../result/p11_{mode}.png")

# p0 = proj[0,:,:]
# p0[p0 > 1.0] = 0.0
# plt.imshow(p0); plt.colorbar(); plt.show()
# plt.savefig("../result/p0_test.png")