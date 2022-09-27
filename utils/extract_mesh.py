import torch
from load_nlos import load_nlos_data
from run_nerf_helpers import *
import scipy.io as scio
import numpy as np
import os, sys
import trimesh
import mcubes

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################################################
name = 'bv_nerf_bmua_new_new'
data = scio.loadmat('./result/' + name + '.mat')
volume = data['bmua_new'][:,:,:]
# print(volume.max())
for i in range(1, 10):
    threshold = round(0.1 * int(i), 1)
    print(threshold)
    vertices, triangles = mcubes.marching_cubes(volume, threshold * volume.max())
    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fill_holes(mesh)
    # mesh.show()
    mesh.export('./result/' + name +'_' + str(threshold) +'.obj')
    grid = 1

