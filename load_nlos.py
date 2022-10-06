import os
# from cv2 import illuminationChange
import numpy as np
import torch
import scipy.io as scio
# import h5py
 
def load_iqi_data(basedir):

    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data_transients'][:,:4096]
    data = data.reshape([int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), -1])
    data = np.transpose(data, [2, 0, 1])

    data = (data[0::2,:,:] + data[1::2,:,:]) / 2 

    illumination_coords = nlos_data['data_illumin_coordinates']
    illumination_coords_x = illumination_coords[0,:].reshape([1,-1])
    illumination_coords_y = np.zeros([1, illumination_coords.shape[1]])
    illumination_coords_z = illumination_coords[1,:].reshape([1,-1])
    camera_grid_positions = np.concatenate([illumination_coords_x, illumination_coords_y, illumination_coords_z], axis=0)

    deltaT = nlos_data['data_time_bin'][0][0] * 3e8

    return data, camera_grid_positions, deltaT

def load_nlos_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    # camera_position = nlos_data['cameraPosition']
    # camera_grid_size = nlos_data['cameraGridSize']
    camera_grid_positions = nlos_data['cameraGridPositions']
    camera_grid_points = nlos_data['cameraGridPoints'][0,:]
    volume_position = nlos_data['hiddenVolumePosition']
    volume_size = nlos_data['hiddenVolumeSize']
    deltaT = nlos_data['deltaT'][0,:][0]

    return data, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT

def load_real_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    # camera_position = nlos_data['cameraPosition']
    # camera_grid_size = nlos_data['cameraGridSize']
    wall_size = nlos_data['wall_size'][0,:][0]
    deltaT = nlos_data['time_bin'][0,:][0] * nlos_data['speed_light'][0,:][0]
    # print(wall_size, deltaT)
    return data, wall_size, deltaT

def load_generated_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    # data = torch.from_numpy(data)

    return data

def load_simtof_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    data = (data[:,:,0::2] + data[:,:,1::2]) / 2 

    deltaT = nlos_data['deltaT'][0][0]
    # data = torch.from_numpy(data)

    return data, deltaT

def load_generated_gt(gtdir):
    volume_gt = scio.loadmat(gtdir)

    volume = volume_gt['Volume']
    xv = volume_gt['x'].reshape([-1])
    yv = volume_gt['y'].reshape([-1])
    zv = volume_gt['z'].reshape([-1])
    volume_vector = [xv,yv,zv]
    return volume, volume_vector