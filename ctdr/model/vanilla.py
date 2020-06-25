import torch
import numpy as np
import torch.nn as nn
import trimesh
from ctdr.utils import util_vis, util_mesh, util_fun
from ctdr.nn import softras_loss
from ctdr.function.rasterizer import dtype

import torch.nn.functional as F

class Model(nn.Module):
    """

    Args:
        labels (int arr, [fx2]) : inward and outward labels for     each face
        mus ( [nmaterials-1]) : we don't save the background material (assumed to be zero)
        mus_fixed_no ( int ) : number of fixed mus.
            if  0, background is 0
            if -1, initialize mus by matrix inversion and set 1
    Returns:


    Examples:
        >>> 

    """
    def __init__(self, ftemplate, proj_geom, nmaterials, mus=None, mus_fixed_no=1, use_disp_param=True, use_center_param=False, wlap=0., wflat=0.):
        super(Model, self).__init__()
        
        self.p_min = 0.
        
        self.nmaterials = nmaterials
        vertices, faces, labels_v, labels_f = util_mesh.load_template(ftemplate)
        
#         print(labels_v.max()+1, nmaterials)
        # assert(labels_v.max()+1 == nmaterials)
        
        self.mus_fixed_no = abs(mus_fixed_no)
        
        # normalize
#         if normalize == 'minmax':
#             vertices = util_mesh.normalize_min_max(vertices)
        
        # register template (not variables)
        self.register_buffer('vertices', torch.tensor(vertices, dtype=dtype))
        self.register_buffer('faces', torch.tensor(faces, dtype=torch.int64))
        
        # register label information for each face
        self.register_buffer('labels', torch.tensor(labels_f, dtype=torch.int32))
        
        self.labels_v = labels_v
        self.labels_v_np = labels_v # for saving later

        
        # -------------------
        # loss
        self.use_lap_loss = True if wlap > 0. else False
        self.use_flat_loss = True if wflat > 0. else False
        self.init_register(mus, mus_fixed_no, use_center_param, use_disp_param)
        
        # if proj_geom['type'][-3:] == 'vec':
        #     from ctdr.nn.fp import FP
        #     self.fp = FP(proj_geom, self.labels)
        #     self.nangles = len(proj_geom['vecs'].shape[0])
        # else:
        from ctdr.nn.fp_opengl import FP
        self.fp = FP(proj_geom, self.labels)
        self.nangles = len(proj_geom['ProjectionAngles'])
        
    #-------------------------------------------------------
    #-------------------------------------------------------
    def init_register(self, mus, mus_fixed_no, use_center_param, use_disp_param=True):
        
        if use_center_param:
            # center as a parameter (rotation can be added as well)
            self.register_parameter('center', \
                nn.Parameter(torch.zeros([1, 3], dtype=dtype) ) )
        else:
            self.register_buffer('center', torch.zeros([1, 3], dtype=dtype))

        # old_code: for each object, use different centers
#             self.register_parameter('center', nn.Parameter(torch.zeros([1, 3*self.nmaterials], dtype=dtype) ) )
#             self.register_buffer('labels_v', torch.tensor(labels_v))
        
        if use_disp_param:
            self.register_parameter('displace', \
            nn.Parameter(torch.zeros(self.vertices.shape, dtype=dtype)))
        else:
            self.register_buffer('displace', \
                                 torch.zeros_like(self.vertices))
            
        if mus_fixed_no == self.nmaterials:
            # mus is not parameter
            self.register_buffer('mus', torch.tensor(mus, dtype=dtype))
        else:
            print('set mu as parameters')
            self.register_parameter('mus', \
                nn.Parameter(torch.tensor(mus, dtype=dtype)))
            
        #------------------------------------------------
        # register loss
        #------------------------------------------------    
        if self.use_lap_loss:
            self.laplacian_loss = \
         softras_loss.LaplacianLoss(self.vertices.cpu(), self.faces.cpu())
        if self.use_flat_loss:
            self.flat_loss = softras_loss.FlattenLoss(self.faces)
        
#         if self.use_flatten_loss:
#             self.flatten_loss = softras_loss.FlattenLoss(self.faces.cpu())
    
    def register_scale(self):
        self.register_parameter('scale', \
                nn.Parameter(torch.ones([1, 3], dtype=dtype)))
    
    def pretransform(self, translation=False, rot=False, ):
        vert = self.vertices
        if translation:
            vert -= self.center
        
        if hasattr(self, 'scale'):
            vert = vert * self.scale
        
        return vert
    
    def deform(self, verts, disp):
        if disp is None: # if disp is the parameter
            vertices = verts + self.displace
                
        else: # if disp is known (NN)
            vertices = verts + disp

#         else:
            #- we can translate for each object but we might lose the fixed topology. Hm:P
            # substract center for each material
#             for m in range(self.nmaterials):
#                 vertices[self.labels_v==m,:] -= \
#                         self.center[:,3*m:3*(m+1)]        
        return vertices
    
    #-------------------------------------------------------
    # forward
    #-------------------------------------------------------
    def forward(self, idx_angles, wedge=0.):
        vertices = self.pretransform(True)
        vertices = self.deform(vertices, None)        
        
        if self.mus_fixed_no == 1:
            self.mus.data[0] = self.p_min
            
        phat, mask_valid = self.fp.forward(vertices, self.faces, idx_angles, self.mus)
        
        edge_loss = util_fun.compute_edge_length(vertices, self.faces) if wedge > 0. else 0.
        lap_loss = 0. if self.use_lap_loss==False else self.laplacian_loss(vertices).mean()
        flat_loss = 0. if self.use_flat_loss==False else self.flat_loss(vertices)
        
        return phat, mask_valid, edge_loss, lap_loss, flat_loss
    
    def forward_neural(self, disp):
        raise NotImplemented
        
    def render(self):
        vertices = self.pretransform(True)
        vertices = self.deform(vertices, None)        
        
        idx_angles = torch.LongTensor(np.arange(self.nangles)).cuda()
            
        phat, mask_valid = self.fp.forward(vertices, self.faces, idx_angles, self.mus)
        return phat, mask_valid
        
    
    def show_params(self):
        out = ""
        
        if hasattr(self, 'scale'):
            out += str(self.scale[0,0].item())
            out += str(self.scale[0,1].item())
            out += str(self.scale[0,2].item())
            
        print(out)
    
    def set_mu0(self, p_cuda):
        self.p_min = p_cuda.min()
        
    
    def estimate_mu(self, p_cuda):
        """
        p_cuda (data) should be in cuda
        """
        print("@ estimate mus")
        self.p_min = p_cuda.min()
        
        if self.nmaterials == 3:
            self.mus.data[0]=self.p_min
            self.mus.data[1]=0
            self.mus.data[2]=1
            out = self.render()
            p2hat = out[0]
            p2mask = out[1]

            self.mus.data[1]=1
            self.mus.data[2]=0
            out = self.render()
            p1hat = out[0]
            p1mask = out[1]

            mask = p2mask * p1mask

            a11 = torch.sum(p1hat*p1hat)
            a22 = torch.sum(p2hat*p2hat)
            a12 = torch.sum(p1hat*p2hat)

            b1 = torch.sum(p1hat*p_cuda)
            b2 = torch.sum(p2hat*p_cuda)

            A = torch.tensor([[a11,a12],[a12,a22]], dtype=dtype)
            b = torch.tensor([b1,b2], dtype=dtype)

            mu_est = torch.pinverse(A).matmul(b)
            self.mus.data[1] = mu_est[0]
            self.mus.data[2] = mu_est[1]
        
        elif self.nmaterials >= 4:
            self.mus.data[0]=self.p_min
            
            for i in range(self.nmaterials-1):
                self.mus.data[i+1] = 0
            
            self.mus.data[-1]=1
            out = self.render()
            p3hat = out[0]
            p3mask = out[1]

            self.mus.data[2]=1
            self.mus.data[3]=0
            out = self.render()
            p2hat = out[0]
            p2mask = out[1]
            
            self.mus.data[1]=1
            self.mus.data[2]=0
            out = self.render()
            p1hat = out[0]
            p1mask = out[1]

            mask = p2mask * p1mask * p3mask

            a11 = torch.sum(p1hat*p1hat)
            a22 = torch.sum(p2hat*p2hat)
            a12 = torch.sum(p1hat*p2hat)
            a13 = torch.sum(p1hat*p3hat)
            a23 = torch.sum(p2hat*p3hat)
            a33 = torch.sum(p3hat*p3hat)
            
            b1 = torch.sum(p1hat*p_cuda)
            b2 = torch.sum(p2hat*p_cuda)
            b3 = torch.sum(p3hat*p_cuda)

            A = torch.tensor([[a11,a12,a13],[a12,a22,a23],[a13,a23,a33]], dtype=dtype)
            b = torch.tensor([b1,b2,b3], dtype=dtype)

            mu_est = torch.pinverse(A).matmul(b)
            self.mus.data[1] = mu_est[0]
            self.mus.data[2] = mu_est[1]
            self.mus.data[3] = mu_est[2]
            
            
#             self.mus[1:3] = mu_est

#         if self.nmaterials == 2:
#             self.mus[0]=0
#             self.mus[1]=1
#             out = self.render()
#             phat = out[0]
#             mask = out[1]

#             a11 = torch.sum(p1hat*p1hat)
#             b1 = torch.sum(p1hat*p_cuda)
            
#             mu_est = b1 / a11

#             mu_est = torch.pinverse(A).matmul(b)
#             self.mus[1:3] = mu_est

    def get_points(self):
        vertices = self.get_displaced_vertices()
        lv = torch.LongTensor(self.labels_v_np)
        
        if self.nmaterials == 2:
            return vertices
        elif self.nmaterials == 3:
            v1 = vertices[lv == 1, :]
            v2 = vertices[lv == 2, :]
            return v1, v2

    def get_displaced_vertices(self):
        vertices = self.pretransform(True)
        vertices = self.deform(vertices, None)
        return vertices

    def get_vf_np(self, return_as_np=True):
        vertices = self.pretransform(True)
        vertices = self.deform(vertices, None).detach()
        
        labels = self.labels.cpu()
        faces = self.faces.cpu()
        
        if return_as_np:
            return vertices.cpu().numpy(), faces.cpu().numpy(), self.labels_v_np, labels.cpu().numpy()
        
