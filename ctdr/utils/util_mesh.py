import numpy as np
import pickle
import trimesh

EPS=1e-10

def normalize_min_max(vertices, min_=-1, max_=1):
    """Normalize np.ndarray vertices
    """
    vertices = (vertices - vertices.min(axis=0))\
        / (vertices.max(axis=0) - vertices.min(axis=0) + EPS)
    
    vertices = vertices*(max_ - min_) + min_
    return vertices

import trimesh
def save_subdivide_mesh(fname, fname_out=None):
    "example:"
    mesh = trimesh.load_mesh(fname)
    v_new, f_new = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    mesh_ = trimesh.Trimesh(v_new, f_new)
    
    if fname_out is None:
        obj_binary = trimesh.exchange.export.export_mesh(mesh_, '../template/sphere_642_1280.obj')

        
def subdivide_mesh_edgesize(fname, fname_out=None, max_edge=2):
    "example:"
    mesh = trimesh.load_mesh(fname)
    v_new, f_new = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge)
    mesh_ = trimesh.Trimesh(v_new, f_new)
    
    if fname_out is None:
        obj_binary = trimesh.exchange.export.export_mesh(mesh_, '../template/sphere_642_1280.obj')

        
def save_mesh(fname, verts, faces, labels_v=None, labels_f=None):
    """
    verts, faces : np array
    """
    if faces.min() == 0:
        faces = faces.copy() + 1
    
    # single object
    if labels_v is None:
        with open(fname, "w") as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for ff in faces:
                f.write(f"f {ff[0]} {ff[1]} {ff[2]}\n")
                
    else:
        nmaterials = labels_v.max()
        
        with open(fname, "w") as f:
            for m in range(nmaterials):
                idx_faces = (labels_f[:,0] == m+1)
                
                in_label = np.unique(labels_f[idx_faces, 0])[0]
                out_label = np.unique(labels_f[idx_faces, 1])[0]
                
                f.write(f"o object{m} {in_label} {out_label}\n")
                
                idx1 = (labels_v == m+1)
                for v in verts[idx1,:]:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
                # check the inward label
                for ff in faces[idx_faces,:]:
                    f.write(f"f {ff[0]} {ff[1]} {ff[2]}\n")

def subdivide_save(fname, verts, faces, lv, lf, max_edge=0.1):
    v_list = []
    f_list = []
    labels_v = []
    labels_f = []

    nmaterials = lv.max()+1
    
    prev_size = -1
    print("before refine: ", verts.shape)

    for i in range(1, nmaterials):
        vi = verts[lv == i, :]
        
#         idxf = np.where(lf[:,0] == i)
        fi = faces[lf[:,0]==i, :]

        fi = fi - fi.min()
        
        # pre-smoothing and subdivide
        print("filter by taubin")
        mesh = trimesh.Trimesh(vi, fi)
        trimesh.smoothing.filter_taubin(mesh, iterations=10)
        # trimesh.smoothing.filter_taubin(mesh, iterations=10)
        # trimesh.smoothing.filter_humphrey(mesh)
        vi, fi = mesh.vertices, mesh.faces
        v_new, f_new = trimesh.remesh.subdivide(vi, fi)

        # The following seems to have serious erros 
        # v_new, f_new = trimesh.remesh.subdivide_to_size(vi, fi, max_edge=max_edge)
        # mesh = trimesh.Trimesh(v_new, f_new)
        # mesh.fix_normals()
        # v_new, f_new = mesh.vertices, mesh.faces
        
        print("f_new.min", f_new.min())
        
        v_list.append(v_new)
        
        if i == 1:
            f_list.append(f_new+1)
        else:
            # i-2 is needed because i begins from 1
            f_list.append(f_new+f_list[i-2].max()+1) 

        #assert vi.shape[0] == fi.max()+1, print(vi.shape[0], fi.max())
            
        labels_v.append(np.ones(v_new.shape[0], dtype=np.int32)*i)

        lft = np.zeros([f_new.shape[0], 2], dtype=np.int32)
        lft[:,0] = i
        lft[:,1] = lf[lf[:,0]==i,1][0]
        labels_f.append(lft)

    print("after refine: ", v_new.shape)
    verts_out = np.concatenate(v_list, 0)
    faces_out = np.concatenate(f_list, 0)
    labels_v_out = np.concatenate(labels_v, 0)
    labels_f_out = np.concatenate(labels_f, 0)

    assert(labels_v_out.max() == labels_f_out.max())
    
    save_mesh(fname, verts_out, faces_out, labels_v_out, labels_f_out)

def load_template(fname):
    """
    load obj file with 
    """
    # example:'o object1 2 3'
    with open(fname) as f:
        lines = f.readlines()
        object_cnt = 0

        labels_v = []
        labels_f = []

        labels_in = []
        labels_out = []
        idx_start_v = [0]
        idx_start_f = [0]
        
        vertices = []
        faces = []

        for line in lines:
            components = line.split()
            if len(components) < 3:
                continue
            
            if components[0:2] == 'v ':
                vertices.append([float(components[1]), float(components[2]), float(components[3])])

            elif components[0:2] == 'f ':
                faces.append([int(components[1]), int(components[2]), int(components[3])])

            elif components[0:2] == 'o ':
                object_cnt += 1
                labels_in.append(int(components[2]))
                labels_out.append(int(components[3]))

                if object_cnt > 1:
                    idx_start_v.append(len(vertices))
                    idx_start_f.append(len(faces))

    vertices = np.array(vertices)
    faces = np.array(faces)
    
    idx_start_v.append(vertices.shape[0])
    idx_start_f.append(faces.shape[0])
    
    labels_f = np.zeros([faces.shape[0], 2], dtype=np.int32)
    labels_v = np.ones(vertices.shape[0], dtype=np.int32)

    if object_cnt == 0:
        # only single material
        labels_v[:] = 1
        labels_f[:, 0] = 1

    else:
        for i in range(object_cnt):
            labels_v[idx_start_v[i]:idx_start_v[i+1]] = i+1
            labels_f[idx_start_f[i]:idx_start_f[i+1], 0] = labels_in[i]
            labels_f[idx_start_f[i]:idx_start_f[i+1], 1] = labels_out[i]

    faces -= 1 # obj file begins from 0
    
    print(f"@statistics of mesh: # of v: {vertices.shape[0]}, f: {faces.shape[0]}")
    return vertices, faces, labels_v, labels_f

def save_obj_two_mesh(fname, mesh1, mesh2):
    """
    mesh1, mesh2: trimesh
    mesh1: outside
    mesh2: inside (face begins from index 1)
    """
    nv_mesh1 = mesh1.vertices.shape[0]
    nf_mesh1 = mesh1.faces.shape[0]
    labels_v = np.zeros(nv_mesh1+mesh2.vertices.shape[0], dtype=np.int32)
    labels_v[:nv_mesh1] = 1

    labels_f = np.zeros([nf_mesh1+mesh2.faces.shape[0], 2], dtype=np.int32)
    labels_f[:nf_mesh1,0] = 1
    labels_f[nf_mesh1:,0] = 2
    labels_f[nf_mesh1:,1] = 1
    labels_v[nv_mesh1:] = 2
    
    verts = np.concatenate([mesh1.vertices, mesh2.vertices], axis=0)
    faces = np.concatenate([mesh1.faces, mesh2.faces+nv_mesh1], axis=0)
    save_mesh(fname, verts, faces, labels_v, labels_f)
    return mesh1, mesh2


def gen_two_sphere(fname, outname, radius=0.3):
    """
    from fname sphere.obj, we make generate sphere outside
    
    Args
        - radius (float) : the radius of inside sphere
    """
    
    # load mesh
    mesh = trimesh.load_mesh(fname)
    
    # normalize to lie [-1,1]
    mesh.vertices = 2*(mesh.vertices -  mesh.vertices.min(0)) / (mesh.vertices.max(0) - mesh.vertices.min(0)) -1
    
    # generate the inside sphere with the given radius
    mesh.vertices *= radius

    nvertices = mesh.vertices.shape[0]
    nfaces = mesh.faces.shape[0]

    verts_new = np.zeros([nvertices*2, 3], dtype=np.float32)
    verts_new[0:nvertices,:] = mesh.vertices
    verts_new[nvertices:,:] = mesh.vertices * 2 # outside sphere

    faces_new = np.zeros([nfaces*2, 3], dtype=np.int32)
    faces_new[0:nfaces,:] = mesh.faces
    faces_new[nfaces:,:] = mesh.faces + nvertices

    new_mesh = trimesh.Trimesh(verts_new, faces_new)
    
    # inward and outward labels
    labels_fx2 = np.zeros([nfaces*2, 2], dtype=np.int32)
    # inside object
    labels_fx2[:nfaces,0] = 2
    labels_fx2[:nfaces,1] = 1
    # outside object
    labels_fx2[nfaces:,0] = 1
    
    labels_v = np.zeros([nvertices*2], dtype=np.int32)
    labels_v[:nvertices] = 0
    labels_v[nvertices:] = 1
    
    np.save(outname[:-4]+"_labels.npy", [labels_v, labels_fx2])
    
    string = new_mesh.export(".obj")
    with open(outname, "w") as f:
        f.write(string)


# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/utils/torus.html#torus

# Make an iterator over the adjacent pairs: (-1, 0), (0, 1), ..., (N - 2, N - 1)
def _make_pair_range(N):
    from itertools import tee
    i, j = tee(range(-1, N))
    next(j, None)
    return zip(i, j)

def torus(
    r: float,
    R: float,
    sides: int,
    rings: int, rotate=False):
    """
    Create vertices and faces for a torus.

    Args:
        r: Inner radius of the torus.
        R: Outer radius of the torus.
        sides: Number of inner divisions.
        rings: Number of outer divisions.
        device: Device on which the outputs will be allocated.

    Returns:
        Meshes object with the generated vertices and faces.
    """
    if not (sides > 0):
        raise ValueError("sides must be > 0.")
    if not (rings > 0):
        raise ValueError("rings must be > 0.")

    verts = []
    for i in range(rings):
        # phi ranges from 0 to 2 pi (rings - 1) / rings
        phi = 2 * np.pi * i / rings
        for j in range(sides):
            # theta ranges from 0 to 2 pi (sides - 1) / sides
            theta = 2 * np.pi * j / sides
            x = (R + r * np.cos(theta)) * np.cos(phi)
            y = (R + r * np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            # This vertex has index i * sides + j
            if rotate:
                verts.append([x, y, z])
            else:
                verts.append([y, z, x])

    faces = []
    for i0, i1 in _make_pair_range(rings):
        index0 = (i0 % rings) * sides
        index1 = (i1 % rings) * sides
        for j0, j1 in _make_pair_range(sides):
            index00 = index0 + (j0 % sides)
            index01 = index0 + (j1 % sides)
            index10 = index1 + (j0 % sides)
            index11 = index1 + (j1 % sides)
            faces.append([index00, index10, index11])
            faces.append([index11, index01, index00])

    return np.array(verts), np.array(faces)+1

def compute_chamfer(file1, file2):
    import kaolin
    mesh1 = kaolin.rep.TriangleMesh.from_obj(file1)
    mesh2 = kaolin.rep.TriangleMesh.from_obj(file2)

    metric = kaolin.metrics.mesh.chamfer_distance(mesh1, mesh2)
    print(metric)
