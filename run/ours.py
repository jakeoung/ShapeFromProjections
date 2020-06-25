import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import h5py
from pymeshfix import _meshfix
import time

import ctdr
from parse_args import args, update_args
from ctdr.model.vanilla import Model
from ctdr.dataset import init_mesh
from ctdr.utils import util_mesh
from ctdr import optimize
import subprocess

#torch.backends.cudnn.benchmark=True

#------------------------------------------------
# load data
#------------------------------------------------
from ctdr.dataset import dataset

# args.data='3auag'
# args.niter=3000
update_args(args)

if args.data.find("tomop") > 0:
    args.nmaterials = int(args.data[-3:-1])+1

ds = dataset.SinoDataset(args.ddata, args.nmaterials, args.eta)
width_physical = ds.proj_geom['DetectorSpacingX']*ds.proj_geom['DetectorColCount']
height_physical = ds.proj_geom['DetectorSpacingY']*ds.proj_geom['DetectorRowCount']
physical_unit = min(width_physical, height_physical)

finit_obj = args.ddata+'/init.obj'

# if os.path.exists(finit_obj) == False:
if True:
    init_mesh.save_init_mesh(finit_obj, args.data, args.nmaterials, physical_unit, args.subdiv)
else:
    print(f"Use existing init file {finit_obj}")

use_center_param = False

mus = np.arange(ds.nmaterials) / (ds.nmaterials-1)
# electorn tomography data
if args.data == '3nanoC':
    #pass
    ds.p -= ds.p.max() * 0.1
    ds.p[ds.p < 0.0] = 0.0
    ds.proj_geom["type"] = "parallel3d"    
    mus = [0, 0.5, 1.2]
    ds.p_original = ds.p.numpy()

#------------------------------------------------
# (optional) guess coarse mesh if args.niter0 > 0
#------------------------------------------------
# find initial guess
nrefine = 1
if args.niter0 < 0:
    args.niter0 *= -1
    if args.niter0 >= 100:
        nrefine = 2
    else:
        nrefine = 3
    
for i in range(nrefine):
    if args.niter0 > 0:
        model = Model(finit_obj, ds.proj_geom, args.nmaterials,
                    mus, args.nmu0, wlap=args.wlap, wflat=args.wflat,
                    use_center_param=use_center_param).cuda()
        
        
        if args.nmu0 <= -1:
            model.estimate_mu(ds.p.cuda())
        else:
            model.set_mu0(ds.p.cuda())

        optimize.run(model, ds, (i+1)*args.niter0, args, i*args.niter0)

        v, f, lv, lf = model.get_vf_np()
        mus = model.mus.detach().cpu().numpy()
        del model.fp
        del model

        if i == 0:
            outresolution = f.shape[0]

        print("@ Save subdivided object and renew")
        finit_obj = args.dresult+'/init_subd.obj'

        if args.data[-1] == "C":
            util_mesh.subdivide_save(finit_obj, v, f, lv, lf)

        # else:
        util_mesh.save_mesh(finit_obj, v, f, lv, lf)

        if args.data[-1] == "A":
            start = time.time()
            tin = _meshfix.PyTMesh()
            tin.load_file(finit_obj)
            _meshfix.clean_from_file(finit_obj, finit_obj)
            tin.clean(max_iters=10, inner_loops=3)
            tin.save_file(finit_obj)
            
            # if i == 0:
            p1 = subprocess.Popen(["./manifold", finit_obj, finit_obj, str(outresolution)])
            p1.communicate()

            elpased_time = time.time() - start
            f = open(f"{args.dresult}/log.txt", "a")
            f.write(f"~ refine_time: {elpased_time:.4f}\n")
            f.close()

#------------------------------------------------
# run the method
#------------------------------------------------
print(finit_obj)
# refine
model = Model(finit_obj, ds.proj_geom, args.nmaterials,
              mus, args.nmu0, wlap=args.wlap, wflat=args.wflat).cuda()
phat = optimize.run(model, ds, args.niter, args, args.niter0*nrefine)

# util_mesh.save_mesh(args.dresult+f'{epoch:04d}.obj', vv.numpy(), ff.numpy(), labels_v, labels_f)

#------------------------------------------------
# save output
#------------------------------------------------
hf = h5py.File(args.dresult+"res.h5", "w")
hf.close()

# for i in range(4):
#     subprocess.Popen(["convert", f"{args.dresult}/*_sino_{i}.png", "-delay", "1000", "-loop", "0" , f"{args.dresult}ani_sino_{i}.gif"])
#     subprocess.Popen(["convert", f"{args.dresult}/*_sino_res_{i}.png", "-delay", "1000", "-loop", "0" , f"{args.dresult}ani_sino_res_{i}.gif"])
#     subprocess.Popen(["convert", f"{args.dresult}/*_render_{i}.png", "-delay", "1000", "-loop", "0" , f"{args.dresult}ani_render_{i}.gif"])
#     subprocess.Popen(["convert", f"{args.dresult}/*_render_m1_{i}.png", "-delay", "1000", "-loop", "0" , f"{args.dresult}ani_render_m1_{i}.gif"])
#     subprocess.Popen(["convert", f"{args.dresult}/*_render_m2_{i}.png", "-delay", "1000", "-loop", "0" , f"{args.dresult}ani_render_m2_{i}.gif"])
