import numpy as np
import pickle
import toml

def sobel_filter(sino):
    """
    sino: [nangle x height x width]
    """
    from skimage import filters
    edges = np.zeros(sino.shape)
    for i in range(sino.shape[0]):
        edges[i,:,:] = filters.roberts(sino[i,:,:])

    return edges

def read_from_toml(fname_toml, nangles=None):    
    with open(fname_toml, 'r') as f:
        dic = toml.load(f)
    
    dic['ProjectionAngles'] = np.array(dic['ProjectionAngles'], dtype=np.float32)
    
    if nangles is not None:
        dic['ProjectionAngles'] = np.linspace(0, np.pi, nangles, False)
        dic['nangles'] = nangles
    
    return dic


def read_proj_info(fname, nangles=None, convert_vec=True):
    """
    fname can be toml (astra proj_geom) or pkl (astra-style vectors)
    """
    
    if fname[-4:] == 'toml':
        proj_geom = read_from_toml(fname, nangles=nangles)
        if proj_geom['type'] == 'cone' and convert_vec:
            proj_geom = geom_2vec(proj_geom)

    elif fname[-3:] == 'pkl':
        with open(fname, 'rb') as f:
            proj_geom = pickle.load(f)
            proj_geom['nangles'] = proj_geom['Vectors'].shape[0]
            
    # if toml and cone beam, c
            
    return proj_geom


# from ASTRA-TOOLBOX
# https://github.com/astra-toolbox/astra-toolbox/blob/10d87f45bc9311c0408e4cacdec587eff8bc37f8/python/astra/creators.py
def create_vol_geom(*varargin):
    """Create a volume geometry structure.
This method can be called in a number of ways:
``create_vol_geom(N)``:
    :returns: A 2D volume geometry of size :math:`N \\times N`.
``create_vol_geom((Y, X))``:
    :returns: A 2D volume geometry of size :math:`Y \\times X`.
``create_vol_geom(Y, X)``:
    :returns: A 2D volume geometry of size :math:`Y \\times X`.
``create_vol_geom(Y, X, minx, maxx, miny, maxy)``:
    :returns: A 2D volume geometry of size :math:`Y \\times X`, windowed as :math:`minx \\leq x \\leq maxx` and :math:`miny \\leq y \\leq maxy`.
``create_vol_geom((Y, X, Z))``:
    :returns: A 3D volume geometry of size :math:`Y \\times X \\times Z`.
``create_vol_geom(Y, X, Z)``:
    :returns: A 3D volume geometry of size :math:`Y \\times X \\times Z`.
``create_vol_geom(Y, X, Z, minx, maxx, miny, maxy, minz, maxz)``:
    :returns: A 3D volume geometry of size :math:`Y \\times X \\times Z`, windowed as :math:`minx \\leq x \\leq maxx` and :math:`miny \\leq y \\leq maxy` and :math:`minz \\leq z \\leq maxz` .
"""
    vol_geom = {'option': {}}
    # astra_create_vol_geom(row_count)
    if len(varargin) == 1 and isinstance(varargin[0], int) == 1:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[0]
    # astra_create_vol_geom([row_count col_count])
    elif len(varargin) == 1 and len(varargin[0]) == 2:
        vol_geom['GridRowCount'] = varargin[0][0]
        vol_geom['GridColCount'] = varargin[0][1]
    # astra_create_vol_geom([row_count col_count slice_count])
    elif len(varargin) == 1 and len(varargin[0]) == 3:
        vol_geom['GridRowCount'] = varargin[0][0]
        vol_geom['GridColCount'] = varargin[0][1]
        vol_geom['GridSliceCount'] = varargin[0][2]
    # astra_create_vol_geom(row_count, col_count)
    elif len(varargin) == 2:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
    # astra_create_vol_geom(row_count, col_count, min_x, max_x, min_y, max_y)
    elif len(varargin) == 6:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['option']['WindowMinX'] = varargin[2]
        vol_geom['option']['WindowMaxX'] = varargin[3]
        vol_geom['option']['WindowMinY'] = varargin[4]
        vol_geom['option']['WindowMaxY'] = varargin[5]
    # astra_create_vol_geom(row_count, col_count, slice_count)
    elif len(varargin) == 3:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['GridSliceCount'] = varargin[2]
    # astra_create_vol_geom(row_count, col_count, slice_count, min_x, max_x, min_y, max_y, min_z, max_z)
    elif len(varargin) == 9:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['GridSliceCount'] = varargin[2]
        vol_geom['option']['WindowMinX'] = varargin[3]
        vol_geom['option']['WindowMaxX'] = varargin[4]
        vol_geom['option']['WindowMinY'] = varargin[5]
        vol_geom['option']['WindowMaxY'] = varargin[6]
        vol_geom['option']['WindowMinZ'] = varargin[7]
        vol_geom['option']['WindowMaxZ'] = varargin[8]

    # set the window options, if not set already.
    if not 'WindowMinX' in vol_geom['option']:
        vol_geom['option']['WindowMinX'] = -vol_geom['GridColCount'] / 2.
        vol_geom['option']['WindowMaxX'] =  vol_geom['GridColCount'] / 2.
        vol_geom['option']['WindowMinY'] = -vol_geom['GridRowCount'] / 2.
        vol_geom['option']['WindowMaxY'] =  vol_geom['GridRowCount'] / 2.
        if 'GridSliceCount' in vol_geom:
            vol_geom['option']['WindowMinZ'] = -vol_geom['GridSliceCount'] / 2.
            vol_geom['option']['WindowMaxZ'] =  vol_geom['GridSliceCount'] / 2.

    return vol_geom


# Modified from ASTRA-TOOLBOX
# https://github.com/astra-toolbox/astra-toolbox/blob/10d87f45bc9311c0408e4cacdec587eff8bc37f8/python/astra/functions.py
def geom_2vec(proj_geom):
    """Returns a vector-based projection geometry from a basic projection geometry.
    :param proj_geom: Projection geometry to convert
    :type proj_geom: :class:`dict`
    """
    if proj_geom['type'] == 'cone':
        angles = proj_geom['ProjectionAngles']
        vectors = np.zeros((len(angles), 12))
        for i in range(len(angles)):
            # source
            vectors[i, 0] = np.sin(angles[i]) * proj_geom['DistanceOriginSource']
            vectors[i, 1] = -np.cos(angles[i]) * proj_geom['DistanceOriginSource']
            vectors[i, 2] = 0

            # center of detector
            vectors[i, 3] = -np.sin(angles[i]) * proj_geom['DistanceOriginDetector']
            vectors[i, 4] = np.cos(angles[i]) * proj_geom['DistanceOriginDetector']
            vectors[i, 5] = 0

            # vector from detector pixel (0,0) to (0,1)
            vectors[i, 6] = np.cos(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 7] = np.sin(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 8] = 0

            # vector from detector pixel (0,0) to (1,0)
            vectors[i, 9] = 0
            vectors[i, 10] = 0
            vectors[i, 11] = proj_geom['DetectorSpacingY']

        proj_geom_out = {}
        proj_geom_out['type'] = 'cone_vec'
        proj_geom_out['DetectorRowCount'] = proj_geom['DetectorRowCount']
        proj_geom_out['DetectorColCount'] = proj_geom['DetectorColCount']
        proj_geom_out['Vectors'] = vectors
        proj_geom_out['DetectorSpacingY'] = proj_geom['DetectorSpacingY']
        proj_geom_out['DetectorSpacingX'] = proj_geom['DetectorSpacingX']

    # PARALLEL
    elif proj_geom['type'] == 'parallel3d':
        angles = proj_geom['ProjectionAngles']
        vectors = np.zeros((len(angles), 12))
        for i in range(len(angles)):

            # ray direction
            vectors[i, 0] = np.sin(angles[i])
            vectors[i, 1] = -np.cos(angles[i])
            vectors[i, 2] = 0

            # center of detector
            vectors[i, 3] = 0
            vectors[i, 4] = 0
            vectors[i, 5] = 0

            # vector from detector pixel (0,0) to (0,1)
            vectors[i, 6] = np.cos(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 7] = np.sin(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 8] = 0

            # vector from detector pixel (0,0) to (1,0)
            vectors[i, 9] = 0
            vectors[i, 10] = 0
            vectors[i, 11] = proj_geom['DetectorSpacingY']
        
        proj_geom_out = {}
        proj_geom_out['type'] = 'parallel3d_vec'
        proj_geom_out['DetectorRowCount'] = proj_geom['DetectorRowCount']
        proj_geom_out['DetectorColCount'] = proj_geom['DetectorColCount']
        proj_geom_out['Vectors'] = vectors
        
    else:
        raise ValueError(
        'No suitable vector geometry found for type: ' + proj_geom['type'])
        
    return proj_geom_out

# def proj2vol(proj_geom, V=256):
#     import astra
#     """
#     for the parallel beam, determine the volume size automatically
#     """
#     max_x = 1.4*(proj_geom['DetectorColCount']*proj_geom['DetectorSpacingX']*0.5)
#     max_z = (proj_geom['DetectorRowCount']*proj_geom['DetectorSpacingY']*0.5)
#     vol_geom = astra.create_vol_geom(V,V,proj_geom['DetectorRowCount'],-max_x,max_x,-max_x,max_x,-max_z,max_z)
    
#     return vol_geom

def proj2vol(proj_geom, V):
    #import astra
    """
    for the parallel beam, determine the volume size automatically
    """
    max_x = 1*(proj_geom['DetectorColCount']*proj_geom['DetectorSpacingX']*0.5)
    max_z = 1*(proj_geom['DetectorRowCount']*proj_geom['DetectorSpacingY']*0.5)
    
    max_coord = max(max_x, max_z)
    
    #if proj_geom['type'].startswith('parallel'):
    #    V = proj_geom
    
    max_x = max_coord
    max_y = max_coord
    max_z = max_coord
    
    vol_geom = create_vol_geom(V, V, V, -max_x,max_x,-max_x,max_x,-max_z,max_z)

    return vol_geom

class AstraTools3D:
    """3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetRows, DetColumns, AnglesVec, ObjSize):
        self.ObjSize = ObjSize
        self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, DetRows, DetColumns, AnglesVec)
        if type(ObjSize) == tuple:
            N1,N2,N3 = [int(i) for i in ObjSize]
        else:
            N1 = N2 = N3 = ObjSize
        self.vol_geom = astra.create_vol_geom(N3, N2, N1)
    def forwproj(self, object3D):
        """Applying forward projection"""
        proj_id, proj_data = astra.create_sino3d_gpu(object3D, self.proj_geom, self.vol_geom)
        astra.data3d.delete(proj_id)
        return proj_data
    def backproj(self, proj_data):
        """Applying backprojection"""
        rec_id, object3D = astra.create_backprojection3d_gpu(proj_data, self.proj_geom, self.vol_geom)
        astra.data3d.delete(rec_id)
        return object3D



def tomophantom2astra(DetRows, DetColumns, AnglesVec, ObjSize):
    self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, DetRows, DetColumns, AnglesVec)
