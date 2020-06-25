import torch
import numpy as np
import os
import h5py
from glob import glob

from torch.utils.data import Dataset, DataLoader
from ctdr.utils import util_sino

def read_foam(fname):
    f = h5py.File(fname, 'r')
    nangles, H, W = f['projs'].shape
    
    proj_geom = {}
    proj_geom['type'] = 'parallel3d'
    proj_geom['ProjectionAngles'] = np.array(f['angs'])
    proj_geom['DetectorColCount'] = W
    proj_geom['DetectorRowCount'] = H
    proj_geom['DetectorSpacingX'] = 2. / W
    proj_geom['DetectorSpacingY'] = 2. / H
    proj_geom['DistanceOriginSource'] = 4.0
    return np.array(f['projs']), proj_geom

def read_mrc(fname, cut_sino):
    import mrcfile
    # read mrc file for FEI extended header
    # angles.mat file contains the angle information
    proj_geom = {}
    
    with mrcfile.open(fname, 'r', permissive=True) as mrc:
        proj = mrc.data[:]
        
        if cut_sino:
            proj = proj[:, cut_sino:-cut_sino, cut_sino:-cut_sino]
        
        proj_geom['type'] = 'parallel3d'
        proj_geom['DetectorColCount'] = proj.shape[2]
        proj_geom['DetectorRowCount'] = proj.shape[1]
        proj_geom['DetectorSpacingX'] = 2. / proj.shape[2]
        proj_geom['DetectorSpacingY'] = 2. / proj.shape[1]
        proj_geom['DistanceOriginSource'] = 4.0
    
    import struct
    proj_geom['ProjectionAngles'] = np.zeros(proj.shape[0])
    
    with open(fname, "rb") as f:
        for i in range(proj.shape[0]):
            f.seek(1024 + (i)*128, 0)
            byte_float = f.read(4)
            proj_geom['ProjectionAngles'][i] = struct.unpack('f', byte_float)[0]*np.pi/180
            
    return proj, proj_geom

class SinoDataset(Dataset):
    def __init__(self, root_dir, nmaterials=2, eta=0, transform=None, dtype=torch.float32):
        """
        Args:
            root_dir (str) : directory name for the dataset containing
                            .h5 file or (sinogram.npy, proj_geom.toml)
            mode (str) : 'np' or 'h5'
        """
        self.nmaterials=nmaterials
        self.eta=eta
        self.p_original = None
        
        # h5
        #h5_list = glob(root_dir+'/*.h5')
        #if len(h5_list) > 0:
        #    fname = h5_list[0]
        #    proj, self.proj_geom = read_foam(fname)
        #    self.p = torch.tensor(proj)
        #    print(fname + 'file load successfuly.')
        #    self.postprocess()
        #    return
        
        # synthetic data
        if os.path.exists(root_dir+'/proj_geom.toml'):
            self.proj_geom = util_sino.read_from_toml(root_dir+'/proj_geom.toml')
            
            try:
                self.p = np.load(root_dir+'/sinogram.npy')
                self.p_original = self.p.copy()
            except:
                raise Exception("Projection data not found")
            
            if self.eta > 0.0:
                self.impose_noise()

            self.p = torch.tensor(self.p, dtype=dtype)
            self.nangles = len(self.proj_geom['ProjectionAngles'])
            self.postprocess()
            return
            
        # vectors (mainly for cone beam)
        elif os.path.exists(root_dir+'/proj_geom.pkl'):
            with open(root_dir+'/proj_geom.pkl', 'rb') as f:
                self.proj_geom = pickle.load(f)
                
            p = np.load(root_dir+'/sinogram.npy')
            self.p = torch.tensor(p, dtype=dtype)
            self.postprocess()
                
        else:
            raise Exception('There is no file of proj_geom.toml or proj_geom.pkl')

    def postprocess(self):
        # self.vol_geom = util_sino.proj2vol(self.proj_geom, V=self.p.shape[1])
        
        if len(self.p.shape) == 3:
            self.nangles = self.p.shape[0]
        else:
            self.nangles = self.p.shape[1]
        
        
    def __len__(self):
        return self.p.shape[0]

    def __getitem__(self, idx):
        return idx, self.p[idx,:]

    def impose_noise(self):
        np.random.seed(1)
        # np.random.randn(self.p_original.shape)

        e_hat = np.random.normal(size=self.p_original.shape)
        relative = np.sqrt( np.sum( ( self.p_original )**2 ) ) / np.sqrt( np.sum( (e_hat ) **2 ) ) 
        self.p = self.p_original + self.eta * e_hat *  relative
        self.p[self.p < 0.0] = 0.0

        print("succesfully impose noise")