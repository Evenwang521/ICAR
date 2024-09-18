import os
import sys
import trimesh
import mesh2sdf
import numpy as np
import time
import skimage, skimage.measure

filename = sys.argv[1] if len(sys.argv) > 1 else  \
    os.path.join(os.path.dirname(__file__), 'data', 'plane.obj')

mesh_scale = 0.8
size = 128
level = 2 / size

mesh = trimesh.load(filename, force='mesh')

# normalize mesh
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale

# fix mesh
t0 = time.time()
sdf, mesh = mesh2sdf.compute(
    vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
t1 = time.time()
sdf = sdf / scale
voxels = sdf.reshape(size, size, size)
vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
mesh.show()

# output
mesh.vertices = mesh.vertices / scale + center
mesh.export(filename[:-4] + '.fixed.obj')
np.save(filename[:-4] + '.npy', sdf)
print('It takes %.4f seconds to process %s' % (t1-t0, filename))
