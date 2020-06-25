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

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    }while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template<typename scalar_t>
__global__ void dr_cuda_backward_color_batch (
    const scalar_t* __restrict__ grad_sino_bxhxwxd, // input dl/ds
    const scalar_t* __restrict__ im_bxhxwxd,
    const int_t_idx* __restrict__ imidx_vf,
    const int_t_idx* __restrict__ cntvisible_bxhxwx1,
    const int_t_idx* __restrict__ firstidx_bxhxwx1,
    const scalar_t* __restrict__ imwei_3vf,
    const scalar_t* __restrict__ points2d_bxfx6,
    const scalar_t* __restrict__ pointslen_bxfx3,
    const int_t_idx* __restrict__ labels_fx2,
    const scalar_t* __restrict__ mus_n,
    const scalar_t* __restrict__ front_facing_bxfx1,
    scalar_t* __restrict__ grad_points2d_bxfx6,
    scalar_t* __restrict__ grad_pointslen_bxfx3,
    scalar_t* __restrict__ grad_mus_n,
    int bnum, int height, int width, int fnum,
    int dnum, int vf, int sanity_check) {

    // bidx * height * width + heiidx * width + wididx
    int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
    int wididx = presentthread % width;
    presentthread = (presentthread - wididx) / width;
    int heiidx = presentthread % height;
    int bidx = (presentthread - heiidx) / height;

    if (bidx >= bnum || heiidx >= height || wididx >= width)
        return;

    // which pixel it belongs to
    const int totalidx1 = bidx * height * width + heiidx * width + wididx;
    const int firstidx = firstidx_bxhxwx1[totalidx1];
    
    // if the pixel has negative value ignore actually.

    // coordinates
    //scalar_t x0 = 2.0 * wididx / width - 1.0;
    //scalar_t y0 = 1.0 - (2.0*heiidx+1.0) / height;
    scalar_t x0 = wididx;
    scalar_t y0 = heiidx;
    
    // in forward pass, we already checked
    //if (sanity_check) {
    //    if (im_bxhxwxd[totalidx1] < -1e-5 || im_bxhxwxd[totalidx1] > MAX_SINO_VAL) 
    //        return;
    //}
    
    // iterate over visible faces
    for (int i=0; i < cntvisible_bxhxwx1[totalidx1]; i++)
    {
        scalar_t sign = 1.0;
        int fidxint = imidx_vf[firstidx+i] - 1;
        if (fidxint < 0) continue;
        
        const int shift1 = bidx * fnum + fidxint;
        const int shift6 = shift1 * 6;
        const int shift3d = shift1 * 3;
        
        if (front_facing_bxfx1[shift1] < 0)
            sign = -1.0;
        
        // See the forward projection
        if (labels_fx2[fidxint*2] == labels_fx2[fidxint*2+1])
            sign *= -1.0;
        
        for (int is_outward=0; is_outward < 2; is_outward++)
        {
            int label = labels_fx2[fidxint*2 + is_outward];
            
            scalar_t mu = mus_n[label];
            
            if (is_outward)
                sign *= -1.0;

            // gradient of L-buffer
            // 3 points in one face
            scalar_t mu_coeff = 0.0;

            const int base_idx = 3*(firstidx + i);
            
            for (int k=0; k < 3; k++) {
                scalar_t w = imwei_3vf[base_idx + k];
                mu_coeff += w * pointslen_bxfx3[shift3d + k];

                int pointshift = shift3d + k;

                // this should be atomic operation
                scalar_t* addr = grad_pointslen_bxfx3 + pointshift;
                scalar_t val = sign * mu * grad_sino_bxhxwxd[totalidx1] * w; // (3), TOCHECK
                atomicAdd(addr, val);
            }

    // gradient of mus
    // remember im_bxhxwxd[totalidx1] += sign * mu * (w0 * r0 + w1 * r1 + w2 * r2);
    // so gradient should be sign * (w0r0 + w1r1 + w2r2)
			
			// background label has the opposite sign
			if (label == 0)
				mu_coeff *= -1.0;
			
            scalar_t val_mu = sign * mu_coeff * grad_sino_bxhxwxd[totalidx1];
            atomicAdd(grad_mus_n + label, val_mu); // we don't save background mu


    // gradient of points
    // here, we calculate dl/dp
    // dl/dp = dldI * dI/dp
    // dI/dp = c0 * dw0 / dp + c1 * dw1 / dp + c2 * dw2 / dp
    // first
    // 4 coorinates
            scalar_t ax = points2d_bxfx6[shift6 + 0];
            scalar_t ay = points2d_bxfx6[shift6 + 1];
            scalar_t bx = points2d_bxfx6[shift6 + 2];
            scalar_t by = points2d_bxfx6[shift6 + 3];
            scalar_t cx = points2d_bxfx6[shift6 + 4];
            scalar_t cy = points2d_bxfx6[shift6 + 5];

            /*
             scalar_t aw = imwei_3vf[totalidx3 + 0];
             scalar_t bw = imwei_3vf[totalidx3 + 1];
             scalar_t cw = imwei_3vf[totalidx3 + 2];

             // use opengl weights!
             scalar_t x = aw * ax + bw * bx + cw * cx;
             scalar_t y = aw * ay + bw * by + cw * cy;
             */

            ////////////////////////////////////////////////////////////////////////////////
            // replace with other variables
            scalar_t m = bx - ax;
            scalar_t p = by - ay;

            scalar_t n = cx - ax;
            scalar_t q = cy - ay;

            scalar_t s = x0 - ax;
            scalar_t t = y0 - ay;
            ///////////////////////////////////////////////////////////////////////////////
            // m* w1 + n * w2 = s
            // p * w1 + q * w2 = t
            // w1 = (sq - nt) / (mq - np)
            // w2 = (mt - sp) / (mq - np)

            scalar_t k1 = s * q - n * t;
            scalar_t k2 = m * t - s * p;
            scalar_t k3 = m * q - n * p;

            /*
             // debug
             // calculate weights
             scalar_t w1 = k1 / (k3 + eps);
             scalar_t w2 = k2 / (k3 + eps);
             scalar_t w0 = 1 - w1 - w2;
             scalar_t ws[3];
             ws[0] = w0;
             ws[1] = w1;
             ws[2] = w2;
             */

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            scalar_t dk1dm = 0;
            scalar_t dk1dn = -t;
            scalar_t dk1dp = 0;
            scalar_t dk1dq = s;
            scalar_t dk1ds = q;
            scalar_t dk1dt = -n;

            scalar_t dk2dm = t;
            scalar_t dk2dn = 0;
            scalar_t dk2dp = -s;
            scalar_t dk2dq = 0;
            scalar_t dk2ds = -p;
            scalar_t dk2dt = m;

            scalar_t dk3dm = q;
            scalar_t dk3dn = -p;
            scalar_t dk3dp = -n;
            scalar_t dk3dq = m;
            scalar_t dk3ds = 0;
            scalar_t dk3dt = 0;

            ///////////////////////////////////////////////////////////////////////////////
            // w1 = k1 / k3
            // w2 = k2 / k3
            // remember we need divide k3 ^ 2
            scalar_t dw1dm = dk1dm * k3 - dk3dm * k1;
            scalar_t dw1dn = dk1dn * k3 - dk3dn * k1;
            scalar_t dw1dp = dk1dp * k3 - dk3dp * k1;
            scalar_t dw1dq = dk1dq * k3 - dk3dq * k1;
            scalar_t dw1ds = dk1ds * k3 - dk3ds * k1;
            scalar_t dw1dt = dk1dt * k3 - dk3dt * k1;

            scalar_t dw2dm = dk2dm * k3 - dk3dm * k2;
            scalar_t dw2dn = dk2dn * k3 - dk3dn * k2;
            scalar_t dw2dp = dk2dp * k3 - dk3dp * k2;
            scalar_t dw2dq = dk2dq * k3 - dk3dq * k2;
            scalar_t dw2ds = dk2ds * k3 - dk3ds * k2;
            scalar_t dw2dt = dk2dt * k3 - dk3dt * k2;

            //////////////////////////////////////////////////////////////////////////////////////
            scalar_t dw1dax = -(dw1dm + dw1dn + dw1ds);
            scalar_t dw1day = -(dw1dp + dw1dq + dw1dt);
            scalar_t dw1dbx = dw1dm;
            scalar_t dw1dby = dw1dp;
            scalar_t dw1dcx = dw1dn;
            scalar_t dw1dcy = dw1dq;

            scalar_t dw2dax = -(dw2dm + dw2dn + dw2ds);
            scalar_t dw2day = -(dw2dp + dw2dq + dw2dt);
            scalar_t dw2dbx = dw2dm;
            scalar_t dw2dby = dw2dp;
            scalar_t dw2dcx = dw2dn;
            scalar_t dw2dcy = dw2dq;

            /*
             scalar_t dw0dax = -(dw1dax + dw2dax);
             scalar_t dw0day = -(dw1day + dw2day);
             scalar_t dw0dbx = -(dw1dbx + dw2dbx);
             scalar_t dw0dby = -(dw1dby + dw2dby);
             scalar_t dw0dcx = -(dw1dcx + dw2dcx);
             scalar_t dw0dcy = -(dw1dcy + dw2dcy);
             */

            // the same color for 3 points
            // thus we can simplify it
            scalar_t c0 = pointslen_bxfx3[shift3d];
            scalar_t c1 = pointslen_bxfx3[shift3d + 1];
            scalar_t c2 = pointslen_bxfx3[shift3d + 2];

            scalar_t dIdax = (c1 - c0) * dw1dax + (c2 - c0) * dw2dax;
            scalar_t dIday = (c1 - c0) * dw1day + (c2 - c0) * dw2day;
            scalar_t dIdbx = (c1 - c0) * dw1dbx + (c2 - c0) * dw2dbx;
            scalar_t dIdby = (c1 - c0) * dw1dby + (c2 - c0) * dw2dby;
            scalar_t dIdcx = (c1 - c0) * dw1dcx + (c2 - c0) * dw2dcx;
            scalar_t dIdcy = (c1 - c0) * dw1dcy + (c2 - c0) * dw2dcy;

            scalar_t dldI = sign * mu * grad_sino_bxhxwxd[totalidx1]
                    / (k3 * k3 + eps); // TOCHECK

            atomicAdd(grad_points2d_bxfx6 + shift6 + 0, dldI * dIdax);
            atomicAdd(grad_points2d_bxfx6 + shift6 + 1, dldI * dIday);

            atomicAdd(grad_points2d_bxfx6 + shift6 + 2, dldI * dIdbx);
            atomicAdd(grad_points2d_bxfx6 + shift6 + 3, dldI * dIdby);

            atomicAdd(grad_points2d_bxfx6 + shift6 + 4, dldI * dIdcx);
            atomicAdd(grad_points2d_bxfx6 + shift6 + 5, dldI * dIdcy);
        }
    }
}

void dr_cuda_backward_batch(at::Tensor grad_sino_bxhxwxd,
        at::Tensor image_bxhxwxd,
        at::Tensor imidx_vf, at::Tensor cntvisible_bxhxwx1,
        at::Tensor firstidx_bxhxwx1,
        at::Tensor imwei_3vf,
        at::Tensor points2d_bxfx6, at::Tensor pointslen_bxfx3,
        at::Tensor labels_fx2, at::Tensor mus_n,
        at::Tensor front_facing_bxfx1,
        at::Tensor grad_points2d_bxfx6, at::Tensor grad_pointslen_bxfx3,
        at::Tensor grad_mus_n) {

    int bnum = grad_sino_bxhxwxd.size(0);
    int height = grad_sino_bxhxwxd.size(1);
    int width = grad_sino_bxhxwxd.size(2);
    int dnum = grad_sino_bxhxwxd.size(3);
    int fnum = grad_points2d_bxfx6.size(1);
    int vf = imidx_vf.size(0);
    
    int sanity_check = 1;

    // for bxhxw image size
    const int threadnum = 1024;
    const int totalthread = bnum * height * width;
    const int blocknum = totalthread / threadnum + 1;

    const dim3 threads(threadnum, 1, 1);
    const dim3 blocks(blocknum, 1, 1);
    

    AT_DISPATCH_FLOATING_TYPES(grad_sino_bxhxwxd.type(),
            "dr_cuda_backward_color_batch", ([&] {
                dr_cuda_backward_color_batch<scalar_t><<<blocks, threads>>>(
                        grad_sino_bxhxwxd.data<scalar_t>(),
                        image_bxhxwxd.data<scalar_t>(),
                        imidx_vf.data<int_t_idx>(),
                        cntvisible_bxhxwx1.data<int_t_idx>(),
                        firstidx_bxhxwx1.data<int_t_idx>(),
                        
                        imwei_3vf.data<scalar_t>(),
                        points2d_bxfx6.data<scalar_t>(),
                        pointslen_bxfx3.data<scalar_t>(),
                        
                        labels_fx2.data<int_t_idx>(),
                        mus_n.data<scalar_t>(),
                        
                        front_facing_bxfx1.data<scalar_t>(),
                        grad_points2d_bxfx6.data<scalar_t>(),
                        grad_pointslen_bxfx3.data<scalar_t>(),
                        grad_mus_n.data<scalar_t>(),
                        bnum, height, width, fnum, dnum, vf, sanity_check);
            }));

    return;
}
