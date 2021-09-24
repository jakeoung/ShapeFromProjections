from torch.autograd import Function
import torch
import sys
sys.path.append('../ctdr/cuda')
import cuda_diff_fp

# dtype=torch.float32
dtype_int=torch.int32

class Rasterizer(Function):
    
    @staticmethod
    def forward(ctx, p_bxfx6, len_bxfx3, front_facing_bxfx1, labels_fx2, mus_n, H, W, max_sino_val):
        """
        p_bxfx6 : projections on 2D plane
        """
        dtype = p_bxfx6.dtype

        B = p_bxfx6.shape[0]
        F = p_bxfx6.shape[1]
        
        # compute bounding box
        p_view_bxfx3x2 = p_bxfx6.view(B, F, 3, 2)
        p_min = torch.min(p_view_bxfx3x2, dim=2)[0]
        p_max = torch.max(p_view_bxfx3x2, dim=2)[0]
        
        # we don't need to check the outside of detector (we will do in cuda)
#         p_min[p_min < 0] = 0
#         p_max[...,0][p_max[...,0] > W-1.] = W-1.
#         p_max[...,1][p_max[...,1] > H-1.] = H-1.

        p_bbox_bxfx4 = torch.cat([p_min, p_max], dim=2)

        # Here, maybe I can impose edge prior

        im_bxhxwx1 = torch.zeros(B, H, W, 1, dtype=dtype).cuda()
        
        cntvisible_bxhxwx1 = torch.zeros(B, H, W, 1, dtype=dtype_int).cuda().contiguous()

        p_bxfx6 = p_bxfx6.contiguous()
        front_facing_bxfx1 = front_facing_bxfx1.contiguous()
        p_bbox_bxfx4 = p_bbox_bxfx4.contiguous()
        len_bxfx3 = len_bxfx3.contiguous()
        
        temp = torch.zeros(0, dtype=dtype).cuda()
        temp_int = torch.zeros(0, dtype=dtype_int).cuda()

        # first compute the number of visible faces for each pixel
        # and compute the projections
        
        cuda_diff_fp.forward(p_bxfx6, front_facing_bxfx1,
        p_bbox_bxfx4, len_bxfx3, labels_fx2, mus_n,
        temp, temp_int, cntvisible_bxhxwx1,
        temp_int,
        im_bxhxwx1, 1, max_sino_val)
        
        ## if backpropagation is not needed, stop
        # return im_bxhxwx1
        
                
        # -----------------------------------------
        # compute the total number of visible faces
        # -----------------------------------------
        cumsum_vf = torch.cumsum(cntvisible_bxhxwx1.reshape(-1), 0, dtype=dtype_int)
        vf = cumsum_vf[-1] # total number of visible faces
        
        # make the output arrays
        imwei_3vf = torch.zeros(3*vf, dtype=dtype).cuda().contiguous()
        idx_vf = torch.zeros(vf, dtype=dtype_int).cuda().contiguous()
        
        # firstidx retrives the first index of 1D array
        firstidx = torch.zeros([B*H*W], dtype=dtype_int).cuda()
        firstidx[1:] = cumsum_vf[:-1].contiguous()
        firstidx_bxhxwx1 = firstidx.reshape(B, H, W, 1).contiguous()
        
        # compute weights for backpropagation
        cuda_diff_fp.forward(p_bxfx6, front_facing_bxfx1,
            p_bbox_bxfx4, len_bxfx3, labels_fx2, mus_n,
            imwei_3vf, idx_vf, cntvisible_bxhxwx1,
            firstidx_bxhxwx1,
            im_bxhxwx1, 0, max_sino_val)
        
        ctx.save_for_backward(im_bxhxwx1, imwei_3vf, idx_vf, 
        cntvisible_bxhxwx1, firstidx_bxhxwx1, p_bxfx6, len_bxfx3, labels_fx2, mus_n, front_facing_bxfx1)
        
        
        return im_bxhxwx1
    
    @staticmethod
    def backward(ctx, dldI_bxhxwx1):
        """
        dldI_bxhxwx1 : gradient w.r.t. image
        dldf         : gradient w.r.t. length feature
        """
        
        im_bxhxwx1, imwei_3vf, imidx_vf, cntvisible_bxhxwx1, firstidx_bxhxwx1, p_bxfx6, len_bxfx3, labels_fx2, mus_n, front_facing_bxfx1 = ctx.saved_tensors
        
        dldp = torch.zeros_like(p_bxfx6).contiguous()
        dldf = torch.zeros_like(len_bxfx3).contiguous()
        dldmu = torch.zeros_like(mus_n).contiguous()
        
        cuda_diff_fp.backward(
             dldI_bxhxwx1.contiguous(), im_bxhxwx1, imidx_vf, cntvisible_bxhxwx1,
            firstidx_bxhxwx1,
            imwei_3vf, p_bxfx6, len_bxfx3, labels_fx2, mus_n, front_facing_bxfx1,
            dldp, dldf, dldmu)

        return dldp, dldf, None, None, dldmu, None, None, None
