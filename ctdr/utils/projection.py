import numpy as np
from scipy.spatial.transform import Rotation

def project_vertices(vertices, angles, source_origin):
    # rotation matrix for view
    M_left = Rotation.from_quat([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]).as_dcm()
    view_vector = np.array([[0,0,-source_origin]])

    M_bx3x3 = compute_M(idangles)
    Mv_bx3xv = np.matmul(M_bx3x3, verts_vx3.transpose())
    Mv_bx3xv = np.matmul(M_left, Mv_bx3xv)
    VMv_bx3xv = Mv_bx3xv + np.expand_dims(view_vector, 2)
    VMv_bxvx3 = VMv_bx3xv.transpose([0,2,1])
    return VMv_bxvx3

def ortho_Mat(left, right, bottom, top, near, far):
#     return np.array(
#     [
#         [2.0 / (right - left), 0, 0, 0],
#         [0, 2.0/(top-bottom), 0, 0 ],
#         [0,0, 2.0/(near-far), 0],
#         [ -(right+left)/(right-left),-(top+bottom)/(top-bottom),-(far+near)/(far-near),1]
#     ], dtype=np.float32)
    return np.array(
    [
        [2.0 / (right - left), 0,0, -(right+left)/(right-left) ],
        [0, 2.0/(top-bottom), 0, -(top+bottom)/(top-bottom) ],
        [0,0, 2.0/(near-far) , -(far+near)/(far-near)],
        [0,0,0,1]
    ], dtype=np.float32)

def perp_Mat_tb(left, right, bottom, top, near, far):
    return np.array(
    [
        [2.0*near / (right - left), 0, (right+left)/(right-left), 0 ],
        [0, 2.0*near/(top-bottom), (top+bottom)/(top-bottom),0 ],
        [0,0, -(far+near)/(far-near) , -2*(far*near)/(far-near)],
        [0,0,-1,0]
    ], dtype=np.float32)

def perp_simple(fovy, ratio, near, far):
    tanfov = np.tan(fovy / 2.0)
    
    mtx = [[1.0 / (ratio * tanfov), 0, 0, 0], \
                [0, 1.0 / tanfov, 0, 0], \
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], \
                [0, 0, -1.0, 0]]
    
    return np.array(mtx, dtype=np.float32)

def perp_Mat(fovy, aspect, zNear, zFar):
    assert(zNear>0)
    assert(zFar>zNear)
    assert(fovy>0)
    top = np.tan(fovy / 2.) * zNear;
    right = top * aspect
    
    return np.array([
        [zNear / right, 0,0,0],
        [0, zNear/top, 0,0],
        [0, 0, -(zFar+zNear) / (zFar - zNear), -2.*zFar*zNear / (zFar-zNear)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def compute_P(proj_geom):
    if proj_geom['type'] == 'parallel3d':
        
        left  = -proj_geom['DetectorSpacingX'] * proj_geom['DetectorColCount'] / 2.0
        bottom = -proj_geom['DetectorSpacingY'] * proj_geom['DetectorRowCount'] / 2.0
        
        halfU = 0.0 * proj_geom['DetectorSpacingX']
        halfV = 0.5 * proj_geom['DetectorSpacingY']
        #halfU, halfV = 0.0, 0.0
        
        diff_max_min_vertices = -4.0*left
                
        P = ortho_Mat(left-halfU, -left-halfU,
                  bottom+halfV, -bottom+halfV,
                  -0.5*diff_max_min_vertices,
                  0.5*diff_max_min_vertices)
        
    elif proj_geom['type'] == 'cone':
        left  = -proj_geom['DetectorSpacingX'] * proj_geom['DetectorColCount'] / 2.0 
        bottom = -proj_geom['DetectorSpacingY'] * proj_geom['DetectorRowCount'] / 2.0
        
        halfU = -0. * proj_geom['DetectorSpacingX']
        halfV = 0 * proj_geom['DetectorSpacingY']
        
        bottom -= 0.5 * proj_geom['DetectorSpacingY']
        
        
        dsum = proj_geom['DistanceOriginSource']+proj_geom['DistanceOriginDetector']
        
        zNear = dsum  # TODO
        zFar  = dsum+3 # it doesn't affect much
        
        P = perp_Mat_tb(left-halfU, -left+halfU, #-4+1, 4-1 = -3,3
                  bottom+halfV, -bottom-halfV,
                  zNear, zFar)
        
        halfV = 0.5 * proj_geom['DetectorSpacingY']
        
        dsum = proj_geom['DistanceOriginSource']+proj_geom['DistanceOriginDetector']
        
        Vhalf = proj_geom['DetectorSpacingY']*proj_geom['DetectorRowCount'] / 2. ;
        aspect = proj_geom['DetectorColCount'] / (float(proj_geom['DetectorRowCount']) )
        zNear = proj_geom['DistanceOriginSource']  # TODO
        zFar  = dsum # it doesn't affect much
        zNear = 1e-8
        zFar = dsum
        
        fovy = 2. * np.math.atan2( Vhalf, dsum )
        #P = perp_Mat(fovy, aspect, zNear, zFar)
        P = perp_simple(fovy, aspect, zNear, zFar)
        
    return P
    
def compute_M(idx_angles, angles):
    """
    Generate model matrix of [B x 3 x 3]
    
    Args:
        - idx_angles (long np [B]) : 
        - angles (float np [num_angles])
    """
    B = idx_angles.shape[0]
    quaternions = np.zeros([B, 4])
    quaternions[:,2] = np.sin(-angles[idx_angles]/2.)
    quaternions[:,3] = np.cos(-angles[idx_angles]/2.)
        
    R_obj_ = Rotation.from_quat(quaternions)
    R_obj  = R_obj_.as_dcm()
                
    return R_obj
