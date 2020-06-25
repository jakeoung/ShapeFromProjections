import numpy as np

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

import tomopy
fname = "../../data/3nanoC/align_shift@tilt.mrc"
p, pg = read_mrc(fname, 10)

prj = np.transpose(p, [0,2,1])
tomopy.align_joint(prj.copy(), pg["ProjectionAngles"])

