import torch
import numpy as np
import torch.nn as nn
from ctdr.function.rasterizer import Rasterizer
from ctdr.utils.projection import compute_P, compute_M
from scipy.spatial.transform import Rotation
from ctdr.nn.fp import get_valid_mask

class FP(nn.Module):
    """Forward projection module

    1. vertex shader
    
    PVM v
    
    where 
    
    P : 
    V : only depends on dist_src for orthographic projection
        [1, 4] matrix
    M : rotate 3D points
    
    Args:
        - proj_geom (dict) : ASTRA-style dictionary
        - labels (int arr, [fx2]) : outward and inward labels for each face

    Returns:


    Examples:
        >>> 
    """
    def __init__(self, proj_geom, labels, dtype=torch.float32):
        super(FP, self).__init__()
        
        self.proj_geom = proj_geom
        self.labels_fx2 = labels.cuda()
        
        # Among PVM, PV part doesn't depend on the viw angle
        # orthographic matrix
        #if proj_geom['type'].startswith('parallel3d'):
        self.max_sino_val = proj_geom["DetectorColCount"]*proj_geom["DetectorSpacingX"]*2
            
        P = compute_P(proj_geom)
        self.P = torch.tensor(P, dtype=dtype).cuda()

        # rotation matrix for view
        # M_left = Rotation.from_quat([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)])
        # self.M_left = torch.tensor(M_left.as_dcm(), dtype=dtype).cuda()
        self.M_left = torch.tensor([[1.0,0,0],[0,0,-1],[0,1,0]], dtype=dtype).cuda()

        # put the camera enough far away from the origin
        view_vector = torch.tensor([[0,0,-self.max_sino_val]], dtype=dtype).cuda()
        self.V_t = view_vector.unsqueeze(2).cuda()

        # to be consistent with OpenGL convention
        self.vec_ray = torch.tensor([0,0,-1], dtype=dtype).cuda()
        
        self.height = self.proj_geom['DetectorRowCount']
        self.width = self.proj_geom['DetectorColCount']
        
        self.rasterize = Rasterizer().apply
        self.angles = torch.tensor(proj_geom['ProjectionAngles'], dtype=dtype).cuda()
        
    def forward(self, verts_vx3, faces_fx3, idx_angles, mus_n):
        """
        Args:
            - verts_vx3 (tensor, float32 [num_pt x 3])
            - faces_fx3 (tensor, long [num_faces x 3])
            - idx_angles (tensor, long [B])
            - mus (tensor, float32 [nmaterials])
        """
        #--------- multiply by VM (to be consistent with astra)
        # rotate with respect to z-axis
        #  [cos (-t), -sin(-t); sin(-t), cos(-t)]
        # =[cos t, sin t; -sint, cost]
        
        # rotate the object = rotate the source with -angle
        #B, v = idx_angles.shape[0], verts_v

        #------------ begin
        # only to make it consistent with ASTRA and OpenGL, 
        # verts_vx3[:,0] *= -1.0
        # verts_vx3[:,2] *= -1.0
        #------------ end

        Mv_bx3xv = verts_vx3.transpose(0,1).unsqueeze(0).repeat([idx_angles.shape[0], 1, 1])
        
        ct = torch.cos(self.angles[idx_angles]).unsqueeze(1).unsqueeze(2)
        st = torch.sin(self.angles[idx_angles]).unsqueeze(1).unsqueeze(2)
        temp0 = Mv_bx3xv[:,0:1,:]*ct + Mv_bx3xv[:,1:2,:]*st
        Mv_bx3xv[:,1:2,:] = -Mv_bx3xv[:,0:1,:]*st + Mv_bx3xv[:,1:2,:]*ct
        Mv_bx3xv[:,0:1,:] = temp0
        
        # old version:
#         idx_np = idx_angles.detach().cpu().numpy()
#         M_np = compute_M(idx_np, self.proj_geom['ProjectionAngles'])
#         M_bx3x3 = torch.FloatTensor(M_np).cuda()
#         M_bx3x3 = torch.tensor(M_np, dtype=dtype).cuda()        
#         Mv_bx3xv = torch.matmul(M_bx3x3, verts_vx3.transpose(0,1))
        
        # view
        Mv_bx3xv = torch.matmul(self.M_left, Mv_bx3xv)
        VMv_bx3xv = Mv_bx3xv + self.V_t
        
        # save the length of vertices to compute attenuation
        len_vert_bxp = torch.norm(VMv_bx3xv, dim=1).unsqueeze(2)
        
        # multiply by P
        VMv_bx4xv = nn.functional.pad(VMv_bx3xv, (0, 0, 0, 1), "constant", 1.0)
        PVMv = torch.matmul(self.P, VMv_bx4xv) # [4 x 4, b x 4 x v], P:[b x ]
        
        proj_bx3xv = PVMv[:, 0:3, :] / (PVMv[:, 3:4, :] + 1e-8)
        proj_bxpx3 = proj_bx3xv.transpose(1,2)
        
        # rearange w.r.t. faces for convenience
        # rearange points
        VMv_bxvx3 = VMv_bx3xv.transpose(1,2)
        
        bvf0 = VMv_bxvx3[:, faces_fx3[:, 0], :] # b f 3
        bvf1 = VMv_bxvx3[:, faces_fx3[:, 1], :]
        bvf2 = VMv_bxvx3[:, faces_fx3[:, 2], :]
#         bverts_bf9 = torch.cat([bvf0, bvf1, bvf2], dim=2)
    
        # rearange lengths
        lf0 = len_vert_bxp[:, faces_fx3[:, 0], :] # shape: [b x f x 3]
        lf1 = len_vert_bxp[:, faces_fx3[:, 1], :]
        lf2 = len_vert_bxp[:, faces_fx3[:, 2], :]
        len_bxfx3 = torch.cat([lf0, lf1, lf2], dim=2)
        
        # rearange 2d point
        xyf0 = proj_bxpx3[:, faces_fx3[:, 0], :2]
        xyf1 = proj_bxpx3[:, faces_fx3[:, 1], :2]
        xyf2 = proj_bxpx3[:, faces_fx3[:, 2], :2]
        proj_bxfx6 = torch.cat([xyf0, xyf1, xyf2], dim=2)
                
        bvf10 = bvf1 - bvf0
        bvf20 = bvf2 - bvf0

        #------------------- compute frontFacing
        # calculate normal for computing front facing later
        normal_bxfx3 = torch.cross(bvf10, bvf20, dim=2)
        
        if self.proj_geom['type'] == 'parallel3d':
            # we rotate the object and the direction of ray is fixed as [0,0,-1]
            # so we only check the z-coordinate
            # the input faces should be declared in counter-clockwise order
            front_facing_bxfx1 = - normal_bxfx3[:,:,2:3]
        else:
            # in the case of cone beam,
            # the source is fixed at (0,0,0) ([0, -1, 0] in astra)
            # we need to compute inner product for all the rays.
            front_facing_bxfx1 = torch.sum(bvf0 * normal_bxfx3, dim=2).unsqueeze(2)
        
        # to be consistent with forward_vecs and avoid numerical errors,
        # change from normalized coordinates [-1,1] to pixel coordinates [0,W]
        proj_bxfx6[:,:,0:5:2] = (proj_bxfx6[:,:,0:5:2]+1.) / 2. * self.width
        proj_bxfx6[:,:,1:6:2] = self.height*(1-proj_bxfx6[:,:,1:6:2]) / 2. - 0.5
        
        self.proj, self.len, self.front_facing = proj_bxfx6, len_bxfx3, front_facing_bxfx1
        
        phat = self.rasterize(proj_bxfx6, len_bxfx3, front_facing_bxfx1,
                             self.labels_fx2, mus_n, self.height, self.width, self.max_sino_val)
        
        phat = phat.squeeze()
        mask_valid = get_valid_mask(phat, self.max_sino_val)
        
        return phat, mask_valid

