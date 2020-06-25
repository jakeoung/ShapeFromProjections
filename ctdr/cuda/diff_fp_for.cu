// Some parts of the codes are based on https://github.com/nv-tlabs/DIB-R
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define eps 1e-15
#define int_t_idx int

template<typename scalar_t>
__global__ void dr_cuda_forward_render_batch(
        const scalar_t* __restrict__ points2d_bxfx6,
        const scalar_t* __restrict__ pointsdirect_bxfx1,
        const scalar_t* __restrict__ pointsbbox_bxfx4,
        const scalar_t* __restrict__ pointslen_bxfx3,
        
        const int_t_idx* __restrict__ labels_fx2,
        const scalar_t* __restrict__ mus_n,
        
        scalar_t* __restrict__ imwei_3vf,
        int_t_idx* __restrict__ imidx_vf,
        int_t_idx* __restrict__ cntvisible_bxhxwx1,
        
        const int_t_idx* __restrict__ firstidx_bxhxwx1,
        
        scalar_t* __restrict__ im_bxhxwxd,
        int bnum, int height, int width, int fnum, int dnum, int vf, int is_the_first_pass,
        int use_mu_fitting, scalar_t max_sino_val) {

    // bidx * height * width + heiidx * width + wididx
    int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

    int wididx = presentthread % width;
    presentthread = (presentthread - wididx) / width;

    int heiidx = presentthread % height;
    int bidx = (presentthread - heiidx) / height;

    if (bidx >= bnum || heiidx >= height || wididx >= width) {
        return;
    }
    
    // we consider bnum, wididx, heiidx

    /////////////////////////////////////////////////////////////////
    // which pixel it belongs to
    const int totalidx1 = bidx * height * width + heiidx * width + wididx;
    
    if (!is_the_first_pass) {
        if (im_bxhxwxd[totalidx1] < -1e-5 || im_bxhxwxd[totalidx1] > max_sino_val) {
            return;
        }
    }
    
    // pixel coordinate
    scalar_t x0 = wididx;
    scalar_t y0 = heiidx;
    
    int cnt_visible = 0;
    // we choose one detector pixel
    // now iterate over all the faces
    ////////////////////////////////////////////////////////////////////////
    for (int fidxint = 0; fidxint < fnum; fidxint++) {

        scalar_t sign = 1.0; // front-facing
        
        // which face it belongs to
        const int shift1 = bidx * fnum + fidxint;
        const int shift4 = shift1 * 4;
        const int shift6 = shift1 * 6;
        const int shift3d = shift1 * 3 * dnum;

        // is the face anti front-facing?
        if (pointsdirect_bxfx1[shift1] < 0) {
             sign = -1.0;
        }
        
        ///////////////////////////////////////////////////////////////
        // will this pixel be influenced by this face?
        scalar_t xmin = pointsbbox_bxfx4[shift4 + 0];
        scalar_t ymin = pointsbbox_bxfx4[shift4 + 1];
        scalar_t xmax = pointsbbox_bxfx4[shift4 + 2];
        scalar_t ymax = pointsbbox_bxfx4[shift4 + 3];

		if (xmin < 0.0 || ymin < 0.0 || xmax > width-1.0 || ymax > height-1.0)
			continue;

        // not covered by this face!
        if (x0 < xmin || x0 > xmax || y0 < ymin || y0 > ymax) {
            continue;
        }

        //////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////
        // this pixel is covered by bbox,
        
        // forward
        scalar_t ax = points2d_bxfx6[shift6 + 0];
        scalar_t ay = points2d_bxfx6[shift6 + 1];
        scalar_t bx = points2d_bxfx6[shift6 + 2];
        scalar_t by = points2d_bxfx6[shift6 + 3];
        scalar_t cx = points2d_bxfx6[shift6 + 4];
        scalar_t cy = points2d_bxfx6[shift6 + 5];

        // replace with other variables
        scalar_t m = bx - ax;
        scalar_t p = by - ay;

        scalar_t n = cx - ax;
        scalar_t q = cy - ay;

        scalar_t s = x0 - ax;
        scalar_t t = y0 - ay;

        // m* w1 + n * w2 = s
        // p * w1 + q * w2 = t
        // w1 = (sq - nt) / (mq - np)
        // w2 = (mt - sp) / (mq - np)

        scalar_t k1 = s * q - n * t;
        scalar_t k2 = m * t - s * p;
        scalar_t k3 = m * q - n * p;

        scalar_t w1 = k1 / (k3 + eps);
        scalar_t w2 = k2 / (k3 + eps);
        scalar_t w0 = 1 - w1 - w2;

        // not lie in the triangle
        if (w0 < 0 || w1 < 0 || w2 < 0) {
            continue;
        }
        
        if (is_the_first_pass) {
            int label_in  = labels_fx2[fidxint*2];
            int label_out = labels_fx2[fidxint*2+1];
            scalar_t mu_in  = mus_n[label_in];
            scalar_t mu_out = mus_n[label_out];
            
            // special case: if an inside object is partially outside, we ignore the inside part
            // we assume that the inside label should be large
            if (label_in < label_out) {
                continue;
            }
                
            // if the two labes are the same,
            // we need to compensate for the intermediate region
            if (label_in == label_out) {
                sign *= -1.0;  
            }
            
            // be careful!
            // the background label has the opposite sign
            // is it really true?
            if (label_out == 0)
				mu_out *= -1.0;
        
            // compute sinogram
            scalar_t r0 = pointslen_bxfx3[shift3d];
            scalar_t r1 = pointslen_bxfx3[shift3d+1];
            scalar_t r2 = pointslen_bxfx3[shift3d+2];
            
            scalar_t bary = (w0 * r0 + w1 * r1 + w2 * r2);
            assert(bary>0);
        
            // find the mu for the face
            // inward label
            im_bxhxwxd[totalidx1] += sign * mu_in * bary;

            // outward label (ignore the background=0) 
            im_bxhxwxd[totalidx1] -= sign * mu_out * bary;
        
            cnt_visible += 1;
            
            if (use_mu_fitting) {
                // phati phatj (3 materials)
                //pipj[3*label_in  + label_in]  += sign * mu_in  * im_bxhxwxd[totalidx1];
                //pipj[3*label_out + label_out] -= sign * mu_out * im_bxhxwxd[totalidx1];
                //pipj[3*label_out + label_in]  +=  mu_in*mu_out* im_bxhxwxd[totalidx1];
                //ppi[label_in]  += p[totalidx1] * mu_in  * im_bxhxwxd[totalidx1];
                //ppi[label_out] += p[totalidx1] * mu_out * im_bxhxwxd[totalidx1];
            }
            
            continue;
        }
        
        // be careful when using firstidx.
        // it is not assigned in the first pass of calculating visible faces
        const int firstidx = firstidx_bxhxwx1[totalidx1];
        const int base_idx = firstidx + cnt_visible;
        
        // retrieve index
        imidx_vf[base_idx] = fidxint + 1;
        
        imwei_3vf[base_idx*3 + 0] = w0;
        imwei_3vf[base_idx*3 + 1] = w1;
        imwei_3vf[base_idx*3 + 2] = w2;
        
        cnt_visible += 1;
    }
    
    if (is_the_first_pass) {
        cntvisible_bxhxwx1[totalidx1] = cnt_visible;
    }
}

void dr_cuda_forward_batch(at::Tensor points2d_bxfx6,
        at::Tensor pointsdirect_bxfx1, at::Tensor pointsbbox_bxfx4,
        at::Tensor pointslen_bxfx3, at::Tensor labels_fx2, at::Tensor mus_n,
        at::Tensor imwei_3vf, at::Tensor imidx_vf,
        at::Tensor cntvisible_bxhxwx1, at::Tensor firstidx_bxhxwx1,
        at::Tensor im_bxhxwxd,
        int is_the_first_pass, float max_sino_val) {

    int bnum = points2d_bxfx6.size(0);
    int fnum = points2d_bxfx6.size(1);
    int height = im_bxhxwxd.size(1);
    int width = im_bxhxwxd.size(2);
    int dnum = im_bxhxwxd.size(3);
    int vf = imidx_vf.size(0);

    int use_mu_fitting = 0;

    // for fxbxhxw image size
    const int threadnum = 1024;
    const int totalthread = bnum * height * width;
    const int blocknum = totalthread / threadnum + 1;

    const dim3 threads(threadnum, 1, 1);
    const dim3 blocks(blocknum, 1, 1);
    
    AT_DISPATCH_FLOATING_TYPES(points2d_bxfx6.type(),
            "dr_cuda_forward_render_batch", ([&] {
                dr_cuda_forward_render_batch<scalar_t><<<blocks, threads>>>(
                        points2d_bxfx6.data<scalar_t>(),
                        pointsdirect_bxfx1.data<scalar_t>(),
                        pointsbbox_bxfx4.data<scalar_t>(),
                        pointslen_bxfx3.data<scalar_t>(),
                        
                        labels_fx2.data<int_t_idx>(),
                        mus_n.data<scalar_t>(),

                        imwei_3vf.data<scalar_t>(),
                        
                        imidx_vf.data<int_t_idx>(),
                        cntvisible_bxhxwx1.data<int_t_idx>(),
                        firstidx_bxhxwx1.data<int_t_idx>(),
                        
                        im_bxhxwxd.data<scalar_t>(),
                        bnum, height, width, fnum, dnum, vf, is_the_first_pass,
                        use_mu_fitting, max_sino_val);
            }));

    return;
}
