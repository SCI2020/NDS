from tkinter import Y
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# from azimuth to 3D vector
def Azimuth_to_vector(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.cat([x,y,z], dim=1)

# Spherical Sampling
def spherical_sample_bin_tensor(camera_grid_positions, r, num_sampling_points):
    [x0,y0,z0] = [camera_grid_positions[0,:],camera_grid_positions[1,:],camera_grid_positions[2,:]]

    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [0,pi]
    theta = torch.linspace(0, np.pi , num_sampling_points).cuda()
    phi = torch.linspace(0, np.pi, num_sampling_points).cuda()
    
    dtheta = (np.pi) / num_sampling_points
    dphi = (np.pi) / num_sampling_points

    grid = torch.stack(torch.meshgrid(r, theta, phi), dim = -1)
    grid_x = torch.stack(torch.meshgrid(x0, theta, phi), dim = -1)
    grid_y = torch.stack(torch.meshgrid(y0, theta, phi), dim = -1)
    grid_z = torch.stack(torch.meshgrid(z0, theta, phi), dim = -1)

    spherical = grid.reshape([-1,3])
    grid_x = grid_x.reshape([-1,3])[:,0]
    grid_y = grid_y.reshape([-1,3])[:,0]
    grid_z = grid_z.reshape([-1,3])[:,0]

    cartesian = spherical2cartesian(spherical)

    cartesian = cartesian + torch.stack([grid_x,grid_y,grid_z], dim=-1)

    # print(spherical[:,1].reshape([-1,1]).max(), spherical[:,1].reshape([-1,1]).min(), spherical[:,2].reshape([-1,1]).max(), spherical[:,2].reshape([-1,1]).min())
    direction = Azimuth_to_vector(spherical[:,1].reshape([-1,1]), spherical[:,2].reshape([-1,1]))
    # direction = spherical[:,1:]
    # return cartesian, dtheta, dphi, theta_max, theta_min, phi_max, phi_min  # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值
    # return cartesian, dtheta, dphi, spherical[:,0]
    return cartesian, direction, spherical[:,1].reshape([-1,1]), spherical[:,2].reshape([-1,1]), dtheta, dphi, r
    
# Spherical Sampling
def spherical_sample_bin_tensor_bbox(camera_grid_positions, r, num_sampling_points, volume_position, volume_size):
    [x0,y0,z0] = [camera_grid_positions[0,:],camera_grid_positions[1,:],camera_grid_positions[2,:]]

    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [0,pi]
    theta = torch.linspace(0, np.pi , num_sampling_points).cuda()
    phi = torch.linspace(0, np.pi, num_sampling_points).cuda()
    
    dtheta = (np.pi) / num_sampling_points
    dphi = (np.pi) / num_sampling_points

    grid = torch.stack(torch.meshgrid(r, theta, phi), dim = -1)
    grid_x = torch.stack(torch.meshgrid(x0, theta, phi), dim = -1)
    grid_y = torch.stack(torch.meshgrid(y0, theta, phi), dim = -1)
    grid_z = torch.stack(torch.meshgrid(z0, theta, phi), dim = -1)

    spherical = grid.reshape([-1,3])
    grid_x = grid_x.reshape([-1,3])[:,0]
    grid_y = grid_y.reshape([-1,3])[:,0]
    grid_z = grid_z.reshape([-1,3])[:,0]

    cartesian = spherical2cartesian(spherical)

    cartesian = cartesian + torch.stack([grid_x,grid_y,grid_z], dim=-1)
    
    # print(spherical[:,1].reshape([-1,1]).max(), spherical[:,1].reshape([-1,1]).min(), spherical[:,2].reshape([-1,1]).max(), spherical[:,2].reshape([-1,1]).min())
    direction = Azimuth_to_vector(spherical[:,1].reshape([-1,1]), spherical[:,2].reshape([-1,1]))
    # direction = spherical[:,1:]
    # return cartesian, dtheta, dphi, theta_max, theta_min, phi_max, phi_min  # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值
    # return cartesian, dtheta, dphi, spherical[:,0]

    box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
    xmin = box_point[0,0]
    xmax = box_point[4,0]
    ymin = box_point[0,1]
    ymax = box_point[2,1]
    zmin = box_point[0,2]
    zmax = box_point[1,2]
    cut_index = (cartesian[:,0]>xmin) * (cartesian[:,0]<xmax) * (cartesian[:,1]>ymin) * (cartesian[:,1]<ymax) * (cartesian[:,2]>zmin) * (cartesian[:,2]<zmax)

    return cartesian, direction, spherical[:,1].reshape([-1,1]), spherical[:,2].reshape([-1,1]), dtheta, dphi, r, cut_index

def encoding(pt, L):
    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    xsin = np.sin(logseq*math.pi*pt[0])
    ysin = np.sin(logseq*math.pi*pt[1])
    zsin = np.sin(logseq*math.pi*pt[2])
    xcos = np.cos(logseq*math.pi*pt[0])
    ycos = np.cos(logseq*math.pi*pt[1])
    zcos = np.cos(logseq*math.pi*pt[2])
    coded_pt = np.reshape(np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos)), (1, 6 * L))

    return coded_pt

def encoding_sph(hist, L, L_view, no_view):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)
    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    logseq_view = np.logspace(start=0, stop=L_view-1, num=L_view, base=2)

    xsin = np.sin((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ysin = np.sin((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zsin = np.sin((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    xcos = np.cos((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ycos = np.cos((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zcos = np.cos((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    if no_view:
        coded_hist = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos), axis=1)
    else:
        thetasin = np.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phisin = np.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        thetacos = np.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phicos = np.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        coded_hist = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos), axis=1)
        # coded_hist = np.concatenate([encoding(hist[k], L) for k in range(hist.shape[0])], axis=0)

    return coded_hist

def encoding_sph_tensor(hist, L, L_view, no_view):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).float().cuda()
    logseq_view = torch.logspace(start=0, end=L_view-1, steps=L_view, base=2).float().cuda()

    xsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ysin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    xcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ycos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    if no_view:
        coded_hist = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos), axis=1)
    else:
        thetasin = torch.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phisin = torch.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        thetacos = torch.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phicos = torch.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        coded_hist = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos), axis=1)
        # coded_hist = np.concatenate([encoding(hist[k], L) for k in range(hist.shape[0])], axis=0)

    return coded_hist

def show_samples(samples,volume_position,volume_size):
    box = volume_box_point(volume_position,volume_size)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(0,0,0,c='g',linewidths=0.03)
    ax.scatter(box[:,0],box[:,1],box[:,2], c = 'b', linewidths=0.03)
    ax.scatter(volume_position[0],volume_position[1],volume_position[2],c = 'b', linewidths=0.03)
    ax.scatter(samples[:,0],samples[:,1],samples[:,2],c='r',linewidths=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('./scatter_samples')
    plt.close()

    plt.scatter(0,0,c='g',linewidths=0.03)
    plt.scatter(box[:,0],box[:,1], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[1],c = 'b', linewidths=0.03)
    plt.scatter(samples[:,0],samples[:,1],c = 'r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.savefig('./scatter_samples_XOY')
    plt.close()

    plt.scatter(0,0,c='g',linewidths=0.03)
    plt.scatter(box[:,0],box[:,2], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[2],c = 'b', linewidths=0.03)
    plt.scatter(samples[:,0],samples[:,2], c = 'r')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.show()
    plt.savefig('./scatter_samples_XOZ')
    plt.close()

    return 0

def volume_box_point(volume_position, volume_size):
    [xv, yv, zv] = [volume_position[0], volume_position[1], volume_position[2]]
    # xv, yv, zv 是物体 volume 的中心坐标
    dx = volume_size[0]
    dy = volume_size[0]
    dz = volume_size[0]
    x = np.concatenate((xv - dx, xv - dx, xv - dx, xv - dx, xv + dx, xv + dx, xv + dx, xv + dx), axis=0).reshape([-1, 1])
    y = np.concatenate((yv - dy, yv - dy, yv + dy, yv + dy, yv - dy, yv - dy, yv + dy, yv + dy), axis=0).reshape([-1, 1])
    z = np.concatenate((zv - dz, zv + dz, zv - dz, zv + dz, zv - dz, zv + dz, zv - dz, zv + dz), axis=0).reshape([-1, 1])
    box = np.concatenate((x, y, z),axis = 1)
    return box

def cartesian2spherical(pt):
    # 函数将直角坐标系下的点转换为球坐标系下的点
    # 输入格式： pt 是一个 N x 3 的 ndarray

    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus

    # spherical_pt[:,2] = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) 
    # spherical_pt[:,2] = spherical_pt[:,2] + (spherical_pt[:,2] > 0).astype(np.int) * (-np.pi)

    return spherical_pt

def spherical2cartesian(pt):
    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0]*torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0]*torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0]*torch.cos(pt[:,1])

    # cartesian_pt = np.zeros(pt.shape)
    # cartesian_pt[:,0] = pt[:,0]*np.sin(pt[:,1]) * np.cos(pt[:,2])
    # cartesian_pt[:,1] = pt[:,0]*np.sin(pt[:,1]) * np.sin(pt[:,2])
    # cartesian_pt[:,2] = pt[:,0]*np.cos(pt[:,1])

    return cartesian_pt

def set_cdt_completekernel_torch(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, zmax, zd, device, n_dipoles = 7):
    xmin = -wall_size / 2
    xmax = wall_size/ 2
    ymin = -wall_size / 2
    ymax = wall_size/ 2
    zmin = 0

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)

    # laser position
    xl = 0
    yl = 0
    zl = 0

    # diffuser positioning
    xd = np.linspace(xmin*2, xmax*2, 2*Nx-1)[None, :, None]
    yd = np.linspace(ymin*2, ymax*2, 2*Ny-1)[None, None, :]
    t = np.linspace(0, 2*zmax, 2*Nz) / c
    t = t[:, None, None]

    #set cdt kernel
    t[0, :] = 1
    d = zd - zl
    z0 = 1 / mu_s
    D = 1 / (3 * (mu_a + mu_s))
    rho = np.sqrt((xd - xl)**2 + (yd - yl)**2)

    # n_dipoles = 7
    ii = np.arange(-n_dipoles, n_dipoles+1)[None, None, :]
    z1 = d * (1 - 2 * ii) - 4*ii*ze - z0
    z2 = d * (1 - 2 * ii) - (4*ii - 2)*ze + z0

    dipole_term = z1 * np.exp(-(z1**2) / (4*D*c*t)) - \
    z2 * np.exp(-(z2**2) / (4*D*c*t))
    dipole_term = np.sum(dipole_term, axis=-1)[..., None]  # sum over dipoles
    diff_kernel = (4*np.pi*D*c)**(-3/2) * t**(-5/2) \
            * np.exp(-mu_a * c * t - rho**2 / (4*D*c*t)) \
            * dipole_term

    psf = torch.from_numpy(diff_kernel.astype(np.float32)).to(device)
    diffusion_psf = psf / torch.sum(psf)
    diffusion_psf = torch.roll(diffusion_psf, (-xd.shape[1]//2+1,-yd.shape[2]//2+1), dims=(1,2))
    diffusion_psf = torch.fft.fftn(diffusion_psf) * torch.fft.fftn(diffusion_psf)
    diffusion_psf = abs(torch.fft.ifftn(diffusion_psf))

    # convert to pytorch and take fft
    diffusion_psf = diffusion_psf[None, None, :, :, :]
    # diffusion_fpsf = diffusion_fpsf.rfft(3, onesided=False)
    diffusion_fpsf = torch.fft.fftn(diffusion_psf, s=(Nz*2,Nx*2-1,Ny*2-1))

    return diffusion_fpsf

# def set_cdt_completekernel_noshift(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, zmax, zd, device, n_dipoles = 7):
def set_cdt_completekernel_noshift(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, zmax, zd, device, n_dipoles = 7):
    xmin = -wall_size / 2
    xmax = wall_size/ 2
    ymin = -wall_size / 2
    ymax = wall_size/ 2
    zmin = 0

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)

    # laser position
    xl = 0
    yl = 0
    zl = 0

    # diffuser positioning
    # xd = np.linspace(xmin*2, xmax*2, 2*Nx-1)[None, :, None]
    # yd = np.linspace(ymin*2, ymax*2, 2*Ny-1)[None, None, :]
    xd = np.linspace(xmin*2, xmax*2, 2*Nx)[None, :, None]
    yd = np.linspace(ymin*2, ymax*2, 2*Ny)[None, None, :]
    # xd = np.linspace(xmin*2, xmax*2, 2*Nx)[None, Nx-kernel_size:Nx+kernel_size, None]
    # yd = np.linspace(ymin*2, ymax*2, 2*Ny)[None, None, Ny-kernel_size:Ny+kernel_size]
    t = np.linspace(0, 2*zmax, 2*Nz) / c
    t = t[:, None, None]

    #set cdt kernel
    t[0, :] = 1
    d = zd - zl
    z0 = 1 / mu_s
    D = 1 / (3 * (mu_a + mu_s))
    rho = np.sqrt((xd - xl)**2 + (yd - yl)**2)

    # n_dipoles = 7
    ii = np.arange(-n_dipoles, n_dipoles+1)[None, None, :]
    z1 = d * (1 - 2 * ii) - 4*ii*ze - z0
    z2 = d * (1 - 2 * ii) - (4*ii - 2)*ze + z0

    dipole_term = z1 * np.exp(-(z1**2) / (4*D*c*t)) - \
    z2 * np.exp(-(z2**2) / (4*D*c*t))
    dipole_term = np.sum(dipole_term, axis=-1)[..., None]  # sum over dipoles
    diff_kernel = (4*np.pi*D*c)**(-3/2) * t**(-5/2) \
            * np.exp(-mu_a * c * t - rho**2 / (4*D*c*t)) \
            * dipole_term

    psf = torch.from_numpy(diff_kernel.astype(np.float32)).to(device)
    diffusion_psf = psf / torch.sum(psf)
    diffusion_psf = torch.roll(diffusion_psf, (-xd.shape[1]//2,-yd.shape[2]//2), dims=(1,2))
    # diffusion_psf = torch.roll(diffusion_psf, (-xd.shape[1]//2+1,-yd.shape[2]//2+1), dims=(1,2))
    diffusion_psf = torch.fft.fftn(diffusion_psf) * torch.fft.fftn(diffusion_psf)
    diffusion_psf = abs(torch.fft.ifftn(diffusion_psf))

    # convert to pytorch and take fft
    diffusion_psf = diffusion_psf[None, None, :, :, :]
    # diffusion_fpsf = diffusion_fpsf.rfft(3, onesided=False)
    diffusion_fpsf = torch.fft.fftn(diffusion_psf, s=(Nz*2,Nx*2,Ny*2))
    return diffusion_fpsf

# if __name__=='__main__': # test for encoding
#     pt = torch.rand(3)
#     coded_pt = encoding(pt, 10)
#     pass

# if __name__ == "__main__": # test for cartesian2spherical
#     x = np.array([1,0,0,1])
#     y = np.array([0,1,0,1])
#     z = np.array([0,0,1,1])
#     pt = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1e-4,-1e-4,1]])
#     spherical_pt = cartesian2spherical(pt)
#     print(spherical_pt)
#     pass

def addnoise(data):
    h,w,t = data.shape
    gau = 0.025 + 0.015 * np.random.randn(h,w,t) + data
    poi = 0.025 * np.random.randn(h,w,t) * gau + gau
    # gau = 0.02 + 0.01 * np.random.randn(h,w,t) + data
    # poi = 0.02 * np.random.randn(h,w,t) * gau + gau
    return poi

def psf_for_nlos(wall_size, deltaT, Nx, Ny, Nz, device):

    width = wall_size / 2
    zmax = Nz*deltaT*2
    slope = width/zmax

    x = np.linspace(-1,1,2*Nx)
    y = np.linspace(-1,1,2*Ny)
    z = np.linspace(0,2,2*Nz)
    grid_z,grid_x,grid_y, = np.meshgrid(z,x,y, indexing='ij')

    psf = abs(((4.*slope)**2)*(grid_x**2 + grid_y**2) - grid_z)
    psf = np.double(psf==np.repeat(np.expand_dims(psf.min(axis=0),axis=0),2*Nz,axis=0))
    psf = psf/np.sum(psf[:,Nx,Ny])
    psf = psf/np.linalg.norm(psf.flatten())
    psf = np.roll(psf, (Nx, Ny), axis = (1,2))
    # psf = torch.from_numpy(psf.astype(float)).cuda()[None, None, :, :, :]
    psf = torch.from_numpy(psf.astype(np.float32)).to(device)
    # psf_t = torch.load('./psf.npy')
    # print(torch.sum((psf-psf_t)**2))
    fpsf = torch.fft.fftn(psf)
    # fpsf = torch.fft.fftn(psf)

    return fpsf

def resamplingOperator(Nz, device):
    # resample matrix
    # mtx = np.zeros([Nz**2,Nz])
    # x = np.linspace(0,Nz**2-1,Nz**2).astype(int)

    # mtx[x,(np.ceil(np.sqrt(x+1))-1).astype(int)] = 1
    # mtx = np.dot(np.diag(1/(np.sqrt(x+1)),k=0),mtx)
    # mtxi = mtx.T

    # K = math.floor(math.log(Nz)/math.log(2))
    # for k in range(K):
    #     mtx  = 0.5*(mtx[::2,:]  + mtx[1:,:][::2,:])
    #     mtxi = 0.5*(mtxi[:,::2] + mtxi[:,1:][:,::2])

    # mtx = torch.from_numpy(mtx.astype(np.float32)).cuda()
    # mtxi = torch.from_numpy(mtxi.astype(np.float32)).cuda()

    # # grid_t = torch.load('./mtx.npy')
    # # print(torch.sum((mtx-grid_t)**2))

    # invmtxi = torch.pinverse(mtxi,1e-1)
    # invmtx = torch.pinverse(mtx,1e-1)

    # return invmtx, invmtxi, mtx, mtxi

    # cuda
    # mtx = torch.zeros([Nz**2,Nz]).cuda()
    # x = torch.linspace(0,Nz**2-1,Nz**2).long().cuda()

    # mtx[x,(torch.ceil(torch.sqrt(x+1))-1).long()] = 1

    # mtx = torch.mm(torch.diag(1/(torch.sqrt(x+1)),diagonal=0),mtx)
    # mtxi = mtx.T

    # K = math.floor(math.log(Nz)/math.log(2))
    # for k in range(K):
    #     mtx  = 0.5*(mtx[::2,:]  + mtx[1:,:][::2,:])
    #     mtxi = 0.5*(mtxi[:,::2] + mtxi[:,1:][:,::2])

    # # mtx = torch.from_numpy(mtx.astype(np.float32)).cuda()
    # # mtxi = torch.from_numpy(mtxi.astype(np.float32)).cuda()

    # # grid_t = torch.load('./mtx.npy')
    # # print(torch.sum((mtx-grid_t)**2))

    # invmtxi = torch.pinverse(mtxi.float(),1e-1)
    # invmtx = torch.pinverse(mtx.float(),1e-1)

    # return invmtx, invmtxi, mtx, mtxi

    # sparse
    # mtx = torch.zeros([Nz**2,Nz], device=device1)
    # x = torch.linspace(0,Nz**2-1,Nz**2).long().to(device1)
    mtx = torch.zeros([Nz**2,Nz], device='cpu')
    x = torch.linspace(0,Nz**2-1,Nz**2).long().cpu()
    mtx[x,(torch.ceil(torch.sqrt(x+1))-1).long()] = 1

    # mtx = mtx.to_sparse()
    # x = torch.diag(1/(torch.sqrt(x+1)),0).to_sparse().to(device1)
    # x = torch.sparse.spdiags(1/(torch.sqrt(x+1)), torch.tensor([0]).to(device1), (Nz**2,Nz**2))
    x = torch.sparse.spdiags(1/(torch.sqrt(x+1)), torch.tensor([0]).cpu(), (Nz**2,Nz**2))

    # x = x.to(device)
    mtx = mtx

    mtx = torch.sparse.mm(x,mtx).to_dense()
    mtxi = mtx.T
    # mtx = mtx
    # mtxi = mtx.to_dense()

    K = math.floor(math.log(Nz)/math.log(2))
    for k in range(K):
        mtx  = 0.5*(mtx[::2,:]  + mtx[1:,:][::2,:])
        mtxi = 0.5*(mtxi[:,::2] + mtxi[:,1:][:,::2])

    invmtxi = torch.pinverse(mtxi.float(),1e-1).to(device)
    invmtx = torch.pinverse(mtx.float(),1e-1).to(device)

    return invmtx, invmtxi, mtx, mtxi

def calculate_reflection_coeff(n1):
    '''
    calculate reflection coefficient given the refractive indices of the
    two materials. This is derived in
    Zhu, J. X., D. J. Pine, and D. A. Weitz.
    "Internal reflection of diffusive light in random media."
    Physical Review A 44.6 (1991): 3948.
    '''
    # integrate to calculate c1 and c2
    theta = np.linspace(0., np.pi/2, 501)
    c1 = abs(np.trapz(fresnel(n1, theta)*np.sin(theta)*np.cos(theta), theta))
    theta = -np.linspace(-np.pi/2, 0., 501)
    c2 = abs(np.trapz(fresnel(n1, theta)*np.sin(theta)*np.cos(theta)**2, theta))
    R = (3*c2 + 2*c1) / (3*c2 - 2*c1 + 2)
    return R

def fresnel(n, theta_i):
    n0 = 1.
    with np.errstate(invalid='ignore'):
        theta_t = np.arcsin(n*np.sin(theta_i) / n0)
        R = 0.5 * (np.sin(theta_i - theta_t)**2 / np.sin(theta_i + theta_t)**2 +
                   np.tan(theta_i - theta_t)**2 / np.tan(theta_i + theta_t)**2)
        R[theta_i == 0] = (n - n0)**2 / (n + n0)**2
        R[np.arcsin(n0/n) < theta_i] = 1.
    return R