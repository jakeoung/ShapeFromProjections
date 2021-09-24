import torch
import numpy as np
import torch.nn as nn
from ctdr.function.rasterizer import Rasterizer
from ctdr.utils.projection import compute_P, compute_M

# MAX_SINO_VAL = 10.0
#MAX_SRC_ORIGIN_PARALLEL = 4.0

def get_valid_mask(phat, max_val):
    mask_pos = (phat > -1e-6)

    mask_valid = mask_pos * (phat <= max_val - 1e-6)
    #mask_valid = mask_neg * (phat <= (MAX_SINO_VAL-10.))

    ndead_pixels = (~mask_pos).sum().item()

    if ndead_pixels >= 1:
        phat_min = phat.min().data.item()
        # mask_neg = ((~mask_pos) * (phat >= -30.))
        mask_neg = (~mask_pos)
        sum_neg = mask_neg.sum().item()

        print("ndead nneg phat.min and max_v:", ndead_pixels, sum_neg, f"{phat_min:.2f}, {phat[mask_valid].max().data.item():.2f}")
        
    return mask_valid


class FP(nn.Module):
    """Forward projection module by astra vectors

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
        
        if self.proj_geom['type'] == 'parallel3d_vec':
            self.max_sino_val = proj_geom["DetectorColCount"]*proj_geom["DetectorSpacingX"]*3
        else:
            self.max_sino_val = proj_geom["DetectorColCount"]*proj_geom["DetectorSpacingX"]*100

        self.vecs = torch.tensor(proj_geom['Vectors'], dtype=dtype).cuda()
#             self.DetS = self.Vectors[:,3:6] - 0.5*proj_geom['DetectorRowCount']*self.Vectors[:,9:12] - 0.5*proj_geom['DetectorColCount']*self.Vectors[:, 6:9]
        
        self.height = self.proj_geom['DetectorRowCount']
        self.width = self.proj_geom['DetectorColCount']
        
        self.rasterize = Rasterizer(self.height, self.width, 0).apply
        
        self.det_pos = self.vecs[:,3:6]
        if self.proj_geom['type'] == 'parallel3d_vec':
            self.det_pos = - self.max_sino_val * self.vecs[:,0:3]

        self.dtype = dtype
            
        # normal vector for the detector plane
        
    def forward(self, verts_vx3, faces_fx3, idx_angles, mus_n):
        # raise NotImplemented()
        #------------------------------------------------
        # compute normal vector for each face
        #------------------------------------------------
        vf0 = verts_vx3[faces_fx3[:, 0], :] # b f 3
        vf1 = verts_vx3[faces_fx3[:, 1], :]
        vf2 = verts_vx3[faces_fx3[:, 2], :]
        
        vf10 = vf1 - vf0
        vf20 = vf2 - vf0
        normal_fx3 = torch.cross(vf10, vf20, dim=1)
        
        #------------------------------------------------
        # project onto 2d detector plane
        #------------------------------------------------
        # input: P, output: p=S+tR
        # n . x + D = 0, D = - n . (detector center)
        # ray vector eq: p= S + t(P-S) = S + tR, R=P-S
        # float t = - (dot(N, S) + D) / dot(N, R);
        # N: detector normal vector
        vecs = self.vecs[idx_angles, :]
        verts_bxvx3 = verts_vx3.unsqueeze(0).repeat(vecs.shape[0],1,1)
        
        # compute N of the detector plane
        # N: towards the origin
        # we already know that a3=0, b1=0, b2=0
        # a1:6. b1:9
        # N_{1} =a_{2}b_{3}-a_{3}b_{2} =  a2b3
        # N_{2} =a_{3}b_{1}-a_{1}b_{3} = -a1b3
        # N_{3} =a_{1}b_{2}-a_{2}b_{1} = 0
        
        v = verts_bxvx3.shape[1]
        
        S_bxvx3 = vecs[:,0:3].unsqueeze(1).expand(-1,v,-1)
#         if self.proj_geom['type'] == 'cone_vec':
#             S_bxvx3 = vecs[:,0:3].unsqueeze(1).expand(-1,v,-1)
#         elif self.proj_geom['type'] == 'parallel3d_vec':
#             S_bxvx3 = verts_bxvx3 + vecs[:,0:3]
        
        N_bx2 = torch.zeros([vecs.shape[0], 2], dtype=self.dtype).cuda() # vecs[:,12]=0
        N_bx2[:,0] =  vecs[:,7]*vecs[:,11]
        N_bx2[:,1] = -vecs[:,6]*vecs[:,11]
        
        if self.proj_geom['type'] == 'parallel3d_vec':
            # for the parallel case:
            # we consider the 3D vertice points as the sources
            # t = (N . S + dist_to_detector_center)
            N_bx2 /= torch.sqrt(N_bx2[:,0:1]**2 + N_bx2[:,1:2]**2)

            R_bxvx3_norm = - S_bxvx3.clone()
            S_bxvx3 = verts_bxvx3
            NS_bxv = torch.sum(N_bx2.unsqueeze(1).expand(-1,v,-1)*S_bxvx3[:,:,:2], dim=2)
            t_bxv = NS_bxv + self.max_sino_val
            
            NR_bxv = torch.sum(N_bx2.unsqueeze(1).expand(-1,v,-1) * R_bxvx3_norm[:,:,0:2], dim=2)
            t_bxv = - t_bxv / ( NR_bxv )
            len_vert_bxvx1 = t_bxv.unsqueeze(2)
            # assert(len_vert_bxvx1.min() > 0.)
            
        else:
            R_bxvx3 = verts_bxvx3 - S_bxvx3
            len_vert_bxvx1 = torch.norm(R_bxvx3, dim=2).unsqueeze(2)

            # to avoid numerical issues, it has the effect of increasing t
            R_bxvx3_norm = R_bxvx3 / torch.norm(R_bxvx3, dim=2, keepdim=True)
            NS_b   = torch.sum(N_bx2 * vecs[:,0:2], dim=1)
            D = - torch.sum(N_bx2 * self.det_pos[:,0:2], dim=1)
            coeff_b = -(NS_b + D)
            NR_bxv = torch.sum(N_bx2.unsqueeze(1).expand(-1,v,-1) * R_bxvx3_norm[:,:,0:2], dim=2)
            t_bxv  = coeff_b.unsqueeze(1) / (NR_bxv+1e-10)

        self.t_bxv = t_bxv
        P_bxvx3 = S_bxvx3 + t_bxv.unsqueeze(2) * R_bxvx3_norm
        
        ## compute u,v coordinates
        # P = detS + u a + v b, where a=vecs[:,6:9], b=vecs[:,9:12]
        # a1 u = P1 - DetS1
        # a2 u = P2 - DetS2
        # b3 v = P3 - DetS3
        
        DetS_bx3 = self.det_pos - 0.5*self.height*vecs[:,9:12] - 0.5*self.width*vecs[:, 6:9]
        P_D_bxvx3 = P_bxvx3 - DetS_bx3.unsqueeze(1).expand(-1,v,-1)
        
        # now we need to change to 2D points
        proj_bxpx2 = torch.zeros([P_D_bxvx3.shape[0],P_D_bxvx3.shape[1], 2], dtype=self.dtype).cuda()
        
        # to avoid nonzero division
        idx1 = torch.abs(vecs[:,6]) > torch.abs(vecs[:,7])
        idx2 = ~idx1
        
        proj_bxpx2[idx1,:,0] = P_D_bxvx3[idx1,:,0] / vecs[idx1,6].unsqueeze(1).expand(-1,v)
        proj_bxpx2[idx2,:,0] = P_D_bxvx3[idx2,:,1] / vecs[idx2,7].unsqueeze(1).expand(-1,v)
        proj_bxpx2[:,:,1]    = P_D_bxvx3[:,:,2] / vecs[:,11].unsqueeze(1).expand(-1,v)        
        
        # rearange 2d point
        xyf0 = proj_bxpx2[:, faces_fx3[:, 0], :]
        xyf1 = proj_bxpx2[:, faces_fx3[:, 1], :]
        xyf2 = proj_bxpx2[:, faces_fx3[:, 2], :]
        proj_bxfx6 = torch.cat([xyf0, xyf1, xyf2], dim=2)

        #-------- compute length
        # rearange lengths
        lf0 = len_vert_bxvx1[:, faces_fx3[:, 0], :] # shape: [b x f x 3]
        lf1 = len_vert_bxvx1[:, faces_fx3[:, 1], :]
        lf2 = len_vert_bxvx1[:, faces_fx3[:, 2], :]
        len_bxfx3 = torch.cat([lf0, lf1, lf2], dim=2)
        
        #-------- compute front_facing
        # R_bxvx3, normal_fx3
        
        if self.proj_geom['type'] == 'cone_vec':
            rbf0 = R_bxvx3_norm[:, faces_fx3[:, 0], :] #bxfx3    
        else:
            rbf0 = vecs[:,0:3].unsqueeze(1).expand(-1,v,-1)[:, faces_fx3[:, 0], :]
        
        front_facing_bxfx1 = torch.sum(rbf0 * normal_fx3.unsqueeze(0).expand(vecs.shape[0],-1,-1), dim=2).unsqueeze(2)
        
        self.proj, self.len, self.front_facing = proj_bxfx6, len_bxfx3, front_facing_bxfx1    
        
        phat = self.rasterize(proj_bxfx6, len_bxfx3, front_facing_bxfx1,
                             self.labels_fx2, mus_n, self.height, self.width, self.max_sino_val)
        
        phat = phat.squeeze()
        mask_valid = get_valid_mask(phat, self.max_sino_val)
        
        return phat, mask_valid