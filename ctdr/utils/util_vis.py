import numpy as np
import matplotlib.pyplot as plt

def convert_to_pyvista_format(f_):
    """
    in PyVista, face supports polygonal data, so we should specificy 3 for the triangle
    """
    f_ = np.array(f_).reshape([-1,3])
    f = np.insert(f_, 0, 3*np.ones(f.shape[0]), axis=1) # for 
    return f

import trimesh
def save_vf_as_img(fname, vertices, faces, nimage=4):
    trim = trimesh.Trimesh(vertices=vertices.data.numpy()*0.4,
                               faces=faces.data.numpy())
    
    
    for i in range(nimage):
        trim.apply_transform(trimesh.transformations.rotation_matrix(0.9*i,[2,1,0]))
        scene = trim.scene()
        try:
            # increment the file name
            file_name = f"{fname[:-4]}_{str(i)}.png"
            # save a render of the object as a png
            png = scene.save_image(resolution=[600, 600], visible=True)
            
            with open(file_name, 'wb') as f:
                f.write(png)
                f.close()
        except BaseException as E:
            print("unable to save image", str(E))

def save_vf_as_img_labels(fname, vertices, faces, labels_v, labels_f):
    nmaterials = labels_v.max()+1
    
    if nmaterials == 2:
        save_vf_as_img(fname, vertices, faces)
        return
        
    for m in range(1,nmaterials):
        vv = vertices[labels_v==m, :]
        l_idx = (labels_f[:, 0] == m)
        ff = faces[l_idx, :]
        ff = ff - ff.min() # make it index from 0
                        
        save_vf_as_img(f"{fname[:-4]}_m{m}.png", vv, ff, 2)
            

def save_sino_as_img(fname, proj, cmap='gray', log=None):
    """
    proj: [nangles, H, W]
    """
    idx1 = proj.shape[0]//4
    idxs = [0, idx1, idx1*2, idx1*3]
    for cnt, i in enumerate(idxs):
        try:
            file_name = f"{fname[:-4]}_{str(cnt)}.png"
            if cmap == 'gray' and cnt != 1:
                plt.imsave(file_name, proj[i,:,:], cmap=cmap)
            
            # save residual
            else:
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.imshow(proj[i,:,:], cmap=cmap)
                plt.colorbar()
                # save only one residual (others are left for paper later)
                # if cnt == 1:
                #     if log == None:
                #         log=f"l2 square mean: {(proj**2).mean()}"
                #     plt.text(30,30,log)
                
                plt.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=-0.01)
                
                plt.clf()
                
            
        except BaseException as E:
            print("unable to save image", str(E))
