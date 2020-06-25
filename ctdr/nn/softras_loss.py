# Soft Rasterizer (SoftRas)
# 
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import numpy as np

from ctdr.function.rasterizer import dtype


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= (laplacian[i, i]+1e-10)

        self.register_buffer('laplacian', torch.tensor(laplacian, dtype=dtype))

    def forward(self, x):
        """
        x [num_points x 3]
        """
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        return x

# normal consistency modified from SoftRas code
class FlattenLoss(nn.Module):
    def __init__(self, faces, max_cos_dihedral=0.8):
        super(FlattenLoss, self).__init__()
        print("initialize flatten loss")
        self.nf = faces.size(0)
        
        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        for v0, v1 in zip(v0s, v1s):
            # the order might doesn't matter as we do inner product
            bface = np.sum(faces == v0, axis=1) + np.sum(faces == v1, axis=1)
            fidxs = np.where(bface == 2)[0]
            
            f1 = faces[fidxs[0]] 
            v2s.append( f1[(f1 != v1) * (f1 != v0)] [0] )
            
            f2 = faces[fidxs[1]] 
            v3s.append( f2[(f2 != v1) * (f2 != v0)] [0] )

        # for v0, v1 in zip(v0s, v1s):
        #     bface = np.sum(faces == v0, axis=1) + np.sum(faces == v1, axis=1)
        #     fidxs = np.where(bface == 2)[0]
            
        #     f1 = faces[fidxs[0]] 
        #     if (np.where(f1 == v0)[0][0] + 1) % 3 == ( np.where(f1 == v1)[0][0] ):
        #         v2s.append( f1[(f1 != v1) * (f1 != v0)] [0] )
        #         f2 = faces[fidxs[1]] 
        #         v3s.append( f2[(f2 != v1) * (f2 != v0)] [0] )

        #     else:
        #         v3s.append( f1[(f1 != v1) * (f1 != v0)] [0] )
        #         f2 = faces[fidxs[1]] 
        #         v2s.append( f2[(f2 != v1) * (f2 != v0)] [0] )

        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')
        
        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

        self.max_cos_dihedral = max_cos_dihedral

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = 1

        v0s = vertices[self.v0s, :]
        v1s = vertices[self.v1s, :]
        v2s = vertices[self.v2s, :]
        v3s = vertices[self.v3s, :]

        c10 = v1s - v0s
        c20 = v2s - v0s
        c30 = v3s - v0s

        n0 = torch.cross(c10, c20)
        n1 = -torch.cross(c10, c30)

        n0n = torch.norm(n0, dim=1)
        n1n = torch.norm(n1, dim=1)

        cos = torch.sum(n0 * n1, dim=1) / ( n0n * n1n )
        # cos_ = cos[cos < 0.70710678118] # 0.52532198881 = cos(45 deg)
        loss = (1.0 - cos).mean()
        return loss
            

class FlattenLossOriginal(nn.Module):
    def __init__(self, faces, max_cos_dihedral=0.8):
        super(FlattenLoss, self).__init__()
        print("initialize flatten loss")
        self.nf = faces.size(0)
        
        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        for v0, v1 in zip(v0s, v1s):
            bface = np.sum(faces == v0, axis=1) + np.sum(faces == v1, axis=1)
            fidxs = np.where(bface == 2)[0]
            
            f1 = faces[fidxs[0]] 
            v2s.append( f1[(f1 != v1) * (f1 != v0)] [0] )
            
            f2 = faces[fidxs[1]] 
            v3s.append( f2[(f2 != v1) * (f2 != v0)] [0] )

        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')
        
        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

        self.max_cos_dihedral = max_cos_dihedral

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = 1

        v0s = vertices[self.v0s, :]
        v1s = vertices[self.v1s, :]
        v2s = vertices[self.v2s, :]
        v3s = vertices[self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)
        
        # ind = cos > self.max_cos_dihedral
        # ndegenerate = ind.sum()
        
        #loss = (cos[ind] + 1.).pow(2).mean()
        # loss = cos[ind].pow(2).mean()
        loss = (1 - cos).mean()
        
        # if ndegenerate > 0:
        #     print("! degenerate dihedral angles", ndegenerate) 
        
        return loss
