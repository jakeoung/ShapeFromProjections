#include <torch/extension.h>

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_DIM3(x, b, h, w, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d), #x " must be same im size")
#define CHECK_DIM2(x, b, f, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == f) && (x.size(2) == d), #x " must be same point size")

////////////////////////////////////////////////////////////
// CUDA forward declarations
void dr_cuda_forward_batch(at::Tensor points2d_bxfx6,
        at::Tensor pointsdirect_bxfx1, at::Tensor pointsbbox_bxfx4,
        at::Tensor pointslen_bxfx3, at::Tensor labels_fx2, at::Tensor mus_n, 
        at::Tensor imwei_3vf, at::Tensor imidx_vf,
        at::Tensor cntvisible_bxhxwx1, at::Tensor firstidx_bxhxwx1,
        at::Tensor im_bxhxwx1,
        int flag_cnt_visible, float max_sino_val);

void dr_forward_batch(at::Tensor points2d_bxfx6,
        at::Tensor pointsdirect_bxfx1, at::Tensor pointsbbox_bxfx4,
        at::Tensor pointslen_bxfx3, at::Tensor labels_fx2, at::Tensor mus_n, 
        at::Tensor imwei_3vf, at::Tensor imidx_vf,
        at::Tensor cntvisible_bxhxwx1, at::Tensor firstidx_bxhxwx1,
        at::Tensor im_bxhxwx1, int flag_cnt_visible, float max_sino_val) {

    CHECK_INPUT(points2d_bxfx6);
    CHECK_INPUT(pointsdirect_bxfx1);
    CHECK_INPUT(pointsbbox_bxfx4);
    CHECK_INPUT(pointslen_bxfx3);
    CHECK_INPUT(labels_fx2);
    CHECK_INPUT(mus_n);

//     CHECK_INPUT(imidx_vf);
    CHECK_INPUT(cntvisible_bxhxwx1);
//     CHECK_INPUT(imwei_3vf);
    CHECK_INPUT(firstidx_bxhxwx1);

    CHECK_INPUT(im_bxhxwx1);
    
    int bnum = points2d_bxfx6.size(0);
    int fnum = points2d_bxfx6.size(1);
    int height = im_bxhxwx1.size(1);
    int width = im_bxhxwx1.size(2);
    int dnum = im_bxhxwx1.size(3);

    CHECK_DIM2(points2d_bxfx6, bnum, fnum, 6);
    CHECK_DIM2(pointsdirect_bxfx1, bnum, fnum, 1);
    CHECK_DIM2(pointsbbox_bxfx4, bnum, fnum, 4);
    CHECK_DIM2(pointslen_bxfx3, bnum, fnum, 3);

//    CHECK_DIM3(imidx_vf, bnum, height, width, mf);
    CHECK_DIM3(cntvisible_bxhxwx1, bnum, height, width, 1);
//     CHECK_DIM3(firstidx_bxhxwx1, bnum, height, width, 1);
//     CHECK_DIM3(imwei_3vf, bnum, height, width, 3*mf);

    CHECK_DIM3(im_bxhxwx1, bnum, height, width, dnum);

    dr_cuda_forward_batch(points2d_bxfx6, pointsdirect_bxfx1,
            pointsbbox_bxfx4, pointslen_bxfx3, 
            labels_fx2, mus_n,
            imwei_3vf, imidx_vf,
            cntvisible_bxhxwx1, firstidx_bxhxwx1,
            im_bxhxwx1, flag_cnt_visible, max_sino_val);
    return;
}

//////////////////////////////////////////////////////////
void dr_cuda_backward_batch(at::Tensor grad_sino_bxhxwxd,
        at::Tensor image_bxhxwxd, at::Tensor imidx_vf,
        at::Tensor cntvisible_bxhxwx1, at::Tensor firstidx_bxhxwx1,
        at::Tensor imwei_3vf, at::Tensor points2d_bxfx6, at::Tensor len_bxfx3,
        at::Tensor labels_fx2, at::Tensor mus_n, at::Tensor front_facing_bxfx1,
        at::Tensor grad_points2d_bxfx6, at::Tensor grad_len_bxfx3, at::Tensor grad_mus_n);

void dr_backward_batch(at::Tensor grad_sino_bxhxwxd, at::Tensor image_bxhxwxd,
        at::Tensor imidx_vf, at::Tensor cntvisible_bxhxwx1,
        at::Tensor firstidx_bxhxwx1,
        at::Tensor imwei_3vf, at::Tensor points2d_bxfx6, at::Tensor len_bxfx3,
        at::Tensor labels_fx2, at::Tensor mus_n,at::Tensor front_facing_bxfx1,
        at::Tensor grad_points2d_bxfx6, at::Tensor grad_len_bxfx3, at::Tensor grad_mus_n) {

    CHECK_INPUT(grad_sino_bxhxwxd);
    CHECK_INPUT(image_bxhxwxd);
    CHECK_INPUT(imidx_vf);
    CHECK_INPUT(cntvisible_bxhxwx1);
    CHECK_INPUT(firstidx_bxhxwx1);
    CHECK_INPUT(imwei_3vf);

    CHECK_INPUT(points2d_bxfx6);
    CHECK_INPUT(len_bxfx3);
    CHECK_INPUT(labels_fx2);
    CHECK_INPUT(mus_n);
    
    CHECK_INPUT(front_facing_bxfx1);
    CHECK_INPUT(grad_points2d_bxfx6);
    CHECK_INPUT(grad_len_bxfx3);
    CHECK_INPUT(grad_mus_n);
    
    int bnum = grad_sino_bxhxwxd.size(0);
    int height = grad_sino_bxhxwxd.size(1);
    int width = grad_sino_bxhxwxd.size(2);
    int dnum = grad_sino_bxhxwxd.size(3);
    int fnum = points2d_bxfx6.size(1);
//     int mf = imidx_vf.size(3);

    CHECK_DIM3(grad_sino_bxhxwxd, bnum, height, width, dnum);
    CHECK_DIM3(image_bxhxwxd, bnum, height, width, dnum);
    
//     CHECK_DIM3(imidx_vf, bnum, height, width, mf);
    CHECK_DIM3(cntvisible_bxhxwx1, bnum, height, width, 1);
//     CHECK_DIM3(imwei_3vf, bnum, height, width, mf*3);

    CHECK_DIM2(points2d_bxfx6, bnum, fnum, 6);
    CHECK_DIM2(len_bxfx3, bnum, fnum, 3);
    CHECK_DIM2(front_facing_bxfx1, bnum, fnum, 1);
    CHECK_DIM2(grad_points2d_bxfx6, bnum, fnum, 6);
    CHECK_DIM2(grad_len_bxfx3, bnum, fnum, 3);

    dr_cuda_backward_batch(grad_sino_bxhxwxd,
            image_bxhxwxd, imidx_vf, cntvisible_bxhxwx1, firstidx_bxhxwx1,
            imwei_3vf, points2d_bxfx6, len_bxfx3,
            labels_fx2, mus_n,  front_facing_bxfx1,
            grad_points2d_bxfx6, grad_len_bxfx3,
            grad_mus_n);

    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dr_forward_batch, "dr forward batch (CUDA)");
    m.def("backward", &dr_backward_batch, "dr backward batch (CUDA)");
}
