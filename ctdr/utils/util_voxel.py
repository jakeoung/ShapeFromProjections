import time
import numpy as np
from trimesh.voxel import ops
import trimesh

import skimage
from skimage import measure
from skimage.filters import threshold_multiotsu, threshold_otsu

def extract_mesh(fname, vol, nmaterials=2, pitch=1.0):
    """
    vol (torch.tensor)
    """
    vol = (vol - vol.min()) / (vol.max()+ - vol.min() + 1e-12)
    
    if nmaterials == 2:
        thresh = threshold_otsu(vol)
        voln = skimage.restoration.denoise_tv_chambolle(voln, 0.0001)

        verts, faces, normals, values = measure.marching_cubes_lewiner(voln, level=thresh, allow_degenerate=False)
        mesh = trimesh.Trimesh(verts, faces)
        obj_ = trimesh.exchange.obj.export_obj(mesh)
        with open(fname, "w") as f:
            f.write(obj_)
        
    else:
        #from sklearn.cluster import KMeans
        #kmeans = KMeans(n_clusters=3).fit(vol.reshape([-1,1]))
        #c = sorted(kmeans.cluster_centers_[:,0])
        #print("centroids of Kmeans:", c)
        
        thresholds = threshold_multiotsu(image, nmaterials)

        
        f = open(fname, 'w')
        vert_start = 0
        
        f.write(f"# attenations by kmeans: {str(c)}")

        for i in range(nmaterials-1):
            f.write(f"o objecect{i} {i+1} {i}")
            
            thresh = (c[i] + c[i+1]) / 2.
            # extract for each material
            print(f"extract a material with a threshold {thresh}")
            new_vol = vol.copy()
            new_vol[new_vol<c[i]]=0.
            new_vol[new_vol>=c[i+1]]=1.
            new_vol = (new_vol - c[i]) / (c[i+1]-c[i] +1e-8)
            
            mesh = ops.matrix_to_marching_cubes(matrix=new_vol, pitch=pitch)
            
            #mesh.export(fname[:-4]+str(i)+".obj")
        
            for vert in mesh.vertices:
                f.write('v %f %f %f\n' % tuple(vert+vert_start))
            
            for face in mesh.faces:
                f.write('f %d %d %d\n' % tuple(face + 1 + vert_start))
                
            vert_start += mesh.vertices.shape[0]
            print(vert_start)
        
        f.close()
        return
    

# def rec_vol(proj_geom, sinogram, vol_geom, algo='fbp', niter=50, mesh_name=None):
    # if algo == 'fbp':
        # rec = init_FBP(proj_geom, sinogram, vol_geom)
    
    # elif algo == 'sirt':
        # rec = init_SIRT(proj_geom, sinogram, vol_geom, niter)
        
    # elif algo == 'tvrdart':
        # rec = init_TVRDART(proj_geom, sinogram, vol_geom, niter)
    
    # elif algo == 'pals':
        # pass
    
    # vol = (rec - rec.min()) / (rec.max() - rec.min())
    
    # mesh = kaolin.conversions.voxelgrid_to_trianglemesh(torch.FloatTensor(vol))
    # tmesh = kaolin.rep.TriangleMesh.from_tensors(mesh[0], mesh[1])
    
    # if mesh_name is not None:
        # tmesh.save_mesh(fname)
        
    # return rec, mesh[0], mesh[1]
