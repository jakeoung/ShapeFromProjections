# import pyrender
# import os
# import glob
# import trimesh
# import matplotlib.pyplot as plt
# import numpy as np

# try:
#     from ctdr.utils import util_mesh
# except:
#     import util_mesh

# def render_from_vf(vert, faces, lv, lf, fname=None):
#     nmaterials = lv.max()

#     base = [[0.3,0.3,0.3,0.2], [0.0, 0.0, 0.9, 0.5], [0.0, 0.9, 0.0, 0.5], [0.9, 0.0, 0.0, 0.7]]
#     mesh = []
    
#     camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
#     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.3],
#                            ambient_light=(0.99, 0.99, 0.99))
    
#     # Set up the light -- a single spot light in the same spot as the camera
#     light = pyrender.SpotLight(color=np.ones(3), intensity=3e2,
#                            innerConeAngle=np.pi/16.0)
#     c = 2**-0.5
#     pose=[[1,0,0,0],[0,c,-c,-2],[0,c,c,2],[0,0,0,1]]
#     scene.add(camera, pose=pose)
    
#     scene.add(light, pose=pose)
    
#     for i in range(nmaterials):
        
#         v = vert[lv==i+1,:]
#         f = faces[lf[:,0]==i+1,:]
#         f = f - f.min()
        
#         if nmaterials > 2:
#             material = pyrender.Material(metallicFactor=0.5,
#             alphaMode='BLEND', is_transparent=True)
#         else:
#             material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.5,
#             #material = None
#             alphaMode='BLEND', baseColorFactor=base[i])
            
#         m = trimesh.Trimesh(v, f)
#         mesh.append( pyrender.Mesh.from_trimesh(m, material=material))
#         scene.add(mesh[i])
        
#     r = pyrender.OffscreenRenderer(viewport_width=480,
#                                 viewport_height=480,
#                                 point_size=2.0)
#     col, dep = r.render(scene)
    
#     if fname is not None:
#         plt.imsave(fname, col)
        
#     return col

# def render_from_obj(fobj):
#     vert, faces, lv, lf = util_mesh.load_template(fobj)
#     return render_from_vf(vert, faces, lv, lf)

# def render(dname):
#     print(f"process {dname}")
#     list_obj = glob.glob(dname+"/*.obj")
#     for fobj in list_obj:
#         print(fobj)
#         col = render_from_obj(fobj)
#         plt.imsave(f"{fobj[:-4]}_blending.png", col)

    
# if __name__ == "__main__":
#     print(os.sys.argv)
#     print("Input data name containing obj files")
#     for d in os.listdir(os.sys.argv[1]):
#         dname = os.sys.argv[1]+'/'+d
#         if os.path.isdir(dname):
#             render(dname)    
# #render(os.sys.argv[1])
