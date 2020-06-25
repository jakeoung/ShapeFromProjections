import torch
import numpy as np
import os
from glob import glob

import trimesh
from ctdr.utils import util_mesh

def make_icosphere(subdivisions, radius):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius = radius) 
    mesh.vertices += 1e-4 # for numerical stability to avoid n.r = 0
    return mesh

def save_obj_meshes_labels(fname, meshes, lf_list):
    """
    meshes: list of meshes
    lf_list: list of labels_f
    """
    nmeshes = len(meshes)
    assert(nmeshes == len(lf_list))

    vidx_list = np.zeros(nmeshes+1, dtype=np.int32)
    fidx_list = np.zeros(nmeshes+1, dtype=np.int32)
    
    verts = meshes[0].vertices
    faces = meshes[0].faces
    for i in range(nmeshes):
        vidx_list[i+1] = vidx_list[i] + meshes[i].vertices.shape[0]
        fidx_list[i+1] = fidx_list[i] + meshes[i].faces.shape[0]
        if i >= 1:
            verts = np.concatenate([verts, meshes[i].vertices], axis=0)
            faces = np.concatenate([faces, meshes[i].faces+vidx_list[i]], axis=0)
        
    labels_v = np.zeros(vidx_list[-1], dtype=np.int32)
    labels_f = np.zeros([fidx_list[-1], 2], dtype=np.int32)
    
    for i in range(nmeshes):
        labels_f[fidx_list[i]:fidx_list[i+1],0:2] = np.array(lf_list[i], dtype=np.int32)
        labels_v[vidx_list[i]:vidx_list[i+1]] = lf_list[i][0] # inside label

    util_mesh.save_mesh(fname, verts, faces, labels_v, labels_f)


def save_init_mesh(fname, data, nmaterials, width_physical, subdivision=3, radius=0.4):
    # 3d object lie between -1 and 1
    radius_ = width_physical*radius
    
    topology_type = 'A'
    if data[-1] >= 'A' and data[-1] < 'Z':
        topology_type = data[-1]
    else:
        raise NotImplementedError
        
    try:
        ncomponents = int(data[-3:-1])
    except:
        ncomponents = 1
    
    mus = []
    
    cc = 0.5* np.array([
        [-1,-1,0],[1, 1,0],
        [0, 0, 1],[0, 0, -1],[0, 0, 0],[1,-1,1],[1, 1, 1],
        [-1, 1., 1],
        
        [1,1,-1],[1,-1,-1]
        ], dtype=np.float32)
    
    if topology_type == 'A':
        if ncomponents == 1:
            mesh = make_icosphere(subdivisions=subdivision, radius = radius_*0.5)
            util_mesh.save_mesh(fname, mesh.vertices+1e-6, mesh.faces)
        elif ncomponents == 2:
            if data=="2kitten02A":
                vv, ff = util_mesh.torus(radius_*0.3, radius_*0.6, 10*subdivision, 10*subdivision, True)
            else:
                vv, ff = util_mesh.torus(radius_*0.3, radius_*0.6, 10*subdivision, 10*subdivision, False)
                # vv[:,2] -= 0.2
            
            util_mesh.save_mesh(fname, vv+1e-6, ff)
        elif ncomponents == 4:
            vv, ff = util_mesh.torus(radius_*0.4, radius_*0.8, 20, 40)
            util_mesh.save_mesh(fname, vv+1e-6, ff)
    
    elif topology_type == 'B':
        # there are many independent components which do not share
        #cc = cc * 0.8
        meshes = []
        lf_list = []
        for i in range(ncomponents):
            mesh = make_icosphere(subdivisions=subdivision, radius=radius_*0.4)
            mesh.vertices -= cc[i]
            meshes.append(mesh)
            lf_list.append( [1,0] )
        save_obj_meshes_labels(fname, meshes, lf_list)    
    
    elif topology_type == 'C':
        """
        Component 1 contains other materials recursively.
        """
        print('@ C type topology detected')
        # outside material
        meshes = []
        meshes.append(make_icosphere(subdivisions=subdivision, radius = radius_*0.6))
        lf_list = []
        lf_list.append([1, 0])
        
        mesh_in1 = make_icosphere(subdivisions=subdivision, radius = radius_*0.3)
        meshes.append(mesh_in1)
        lf_list.append([2, 1])
        
        if nmaterials == 3:
            pass
            
        elif nmaterials == 4:
            mesh_in2 = make_icosphere(subdivisions=subdivision, radius = radius_*0.2)
            meshes.append(mesh_in2)
            lf_list.append([3, 2])
            
        
        else:
            raise NotImplementedError
        
        save_obj_meshes_labels(fname, meshes, lf_list)

    elif topology_type == 'D':
        # outside one object and inside is like B
        meshes = []
        mesh1 = make_icosphere(subdivisions=subdivision, radius = radius_*0.99)
        mesh1.vertices += 1e-5
        meshes.append(mesh1)
        lf_list = []
        lf_list.append([1, 0])
        
        ncomponents = int(data[-3:-1])
        
        if nmaterials <= 3:
            for i in range(ncomponents-1):
                mesh = make_icosphere(subdivisions=subdivision-1, radius=radius_*0.2)
                mesh.vertices -= cc[i]
                meshes.append(mesh)
                lf_list.append( [2,1] )

        elif nmaterials >= 4:
            # in case all the inside objects are different
            for i in range(ncomponents-1):
                mesh = make_icosphere(subdivisions=subdivision, radius=radius_*0.1)
                mesh.vertices -= cc[i]
                meshes.append(mesh)
                lf_list.append( [i+2,1] )

            
        save_obj_meshes_labels(fname, meshes, lf_list)    
        

# forward(ctx, p_bxfx6, len_bxfx3, front_facing_bxfx1, labels_fx2, mus_n, H, W):
