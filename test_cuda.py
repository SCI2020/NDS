import numpy
import torch
import numpy as np
# from ncrelu_cuda import ncrelu_forward_cuda
import ncrelu_cuda
from run_netf_helpers import *
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')

sampling_points = 33
batch = 1024
# camera_grid_positions = torch.zeros(3, 4).cuda()
camera_grid_positions = torch.randn(3, batch).cuda()
# camera_grid_positions = torch.ones(3, 4).cuda()
# camera_grid_positions[0,:] = 1
# camera_grid_positions[1,:] = 2
# camera_grid_positions[2,:] = 3
r = torch.randn(batch).cuda()
# r = torch.ones(batch).cuda()
time0 = time.time()
res_cuda, res_cuda_dir = ncrelu_cuda.ncrelu_forward_cuda(camera_grid_positions, r, sampling_points)
print(time.time()-time0)
print(res_cuda.shape, res_cuda_dir.shape)

# # print(f'{camera_grid_positions[0, 0]}, {camera_grid_positions[0, 1]}, {camera_grid_positions[0, 2]}')
# # print(f'b: {b.shape}')
# # print(camera_grid_positions)
# # print(b)
# # print(f'{b[0, 0]}, {b[0, 1]}, {b[0, 2]}')

time0 = time.time()
theta = torch.linspace(0, np.pi, sampling_points).cuda()
phi = torch.linspace(0, np.pi, sampling_points).cuda()
res_theta, res_phi = torch.meshgrid(theta, phi)
res_theta = res_theta.reshape([-1, 1])
res_phi = res_phi.reshape([-1, 1])

# print(r.shape, theta.shape, phi.shape)

grid = torch.stack(torch.meshgrid(r, theta, phi), dim = -1)
# print(time.time()-time0)
print(grid.shape)

spherical = grid.reshape([-1,3])
# print(spherical.device)
cartesian = spherical2cartesian(spherical)
# print(time.time()-time0)

[x0,y0,z0] = [camera_grid_positions[0,:],camera_grid_positions[1,:],camera_grid_positions[2,:]]
grid_x = torch.stack(torch.meshgrid(x0, theta, phi), dim = -1)
grid_y = torch.stack(torch.meshgrid(y0, theta, phi), dim = -1)
grid_z = torch.stack(torch.meshgrid(z0, theta, phi), dim = -1)
grid_x = grid_x.reshape([-1,3])[:,0]
grid_y = grid_y.reshape([-1,3])[:,0]
grid_z = grid_z.reshape([-1,3])[:,0]
cartesian = cartesian + torch.stack([grid_x,grid_y,grid_z], dim=-1)
# print(time.time()-time0)
# print(cartesian.device)

direction = Azimuth_to_vector(spherical[:,1].reshape([-1,1]), spherical[:,2].reshape([-1,1]))
print(direction.shape)
print(time.time()-time0)

cartesian = cartesian.reshape(batch,sampling_points,sampling_points,3)
direction = direction.reshape(batch,sampling_points,sampling_points,3)
# print(cartesian.shape)

# diff = res_cuda - grid.permute([3,0,1,2])
diff = res_cuda - cartesian.permute([3,0,1,2])
print(diff.max(), diff.min())

diff = res_cuda_dir - direction.permute([3,0,1,2])
print(diff.max(), diff.min())

# # grid_x = torch.stack(torch.meshgrid(x0, theta, phi), dim = -1)
# # grid_y = torch.stack(torch.meshgrid(y0, theta, phi), dim = -1)
# # grid_z = torch.stack(torch.meshgrid(z0, theta, phi), dim = -1)

# spherical = grid.reshape([-1,3])
# # grid_x = grid_x.reshape([-1,3])[:,0]
# # grid_y = grid_y.reshape([-1,3])[:,0]
# # grid_z = grid_z.reshape([-1,3])[:,0]

# cartesian = spherical2cartesian(spherical)
# print(cartesian.shape)


# cartesian = cartesian + torch.stack([grid_x,grid_y,grid_z], dim=-1)

# # print(spherical[:,1].reshape([-1,1]).max(), spherical[:,1].reshape([-1,1]).min(), spherical[:,2].reshape([-1,1]).max(), spherical[:,2].reshape([-1,1]).min())
# direction = Azimuth_to_vector(spherical[:,1].reshape([-1,1]), spherical[:,2].reshape([-1,1]))



# res_theta_cuda = res_cuda[0, :, :, 0]
# res_phi_cuda = res_cuda[0, :, :, 1]


########################################################
# # print(res_theta_cuda.squeeze())
# # print(res_theta.squeeze())
# # print(res_phi_cuda.squeeze())
# # print(res_phi.squeeze())


# print(res_cuda[:,0,...].reshape([3,sampling_points*sampling_points]))
# print(res_cuda[:,1,...].reshape([3,sampling_points*sampling_points]))
# print(res_cuda[:,2,...].reshape([3,sampling_points*sampling_points]))
# print(res_cuda[:,3,...].reshape([3,sampling_points*sampling_points]))

# print(grid.permute([3,0,1,2])[:,0,...].reshape([3, sampling_points*sampling_points]))
# print(cartesian.permute([3,0,1,2])[:,0,...].reshape([3, sampling_points*sampling_points]))

# print(cartesian.permute([3,0,1,2])[:,0,...].reshape([3, sampling_points*sampling_points])-res_cuda[:,0,...].reshape([3,sampling_points*sampling_points]))
# print(res_theta.squeeze())
# print(res_phi.squeeze())