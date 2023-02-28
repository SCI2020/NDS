import numpy as np
import scipy.io as scio
import pdb
# import h5py

##############################################
# data: [time, spatial]
# camera_grid_positions: [time, spatial]
# deltaT: metre
# wall_size: metre
##############################################

def load_iqi_data(basedir):

    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data_transients'][:,:4096]
    data = np.transpose(data, [1, 0])
    # data = (data[0::2,:] + data[1::2,:]) / 2 
    # data = data.reshape([int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), -1])
    # data = np.transpose(data, [2, 0, 1])
    # data = (data[0::2,:,:] + data[1::2,:,:]) / 2 

    illumination_coords = nlos_data['data_illumin_coordinates']
    illumination_coords_x = illumination_coords[0,:].reshape([1,-1])
    illumination_coords_y = np.zeros([1, illumination_coords.shape[1]])
    illumination_coords_z = illumination_coords[1,:].reshape([1,-1])
    camera_grid_positions = np.concatenate([illumination_coords_x, illumination_coords_y, illumination_coords_z], axis=0)

    deltaT = nlos_data['data_time_bin'][0][0] * 3e8
    wall_size = camera_grid_positions[0,:].max() * 2

    mask = nlos_data['parameters_wall_visible']
    mask = (mask.squeeze() == 1)
    # print(mask.shape)
    # print(data.shape, camera_grid_positions.shape)
    # print(data[:, mask].shape, camera_grid_positions[:, mask].shape)
    data = data[:, mask]
    camera_grid_positions = camera_grid_positions[:, mask]
    return data, camera_grid_positions, deltaT, wall_size

def load_iqi_los_data(basedir):

    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data_los_transients'][:,:4096]
    data = np.transpose(data, [1, 0])
    # data = (data[0::2,:] + data[1::2,:]) / 2 
    # data = data.reshape([int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), -1])
    # data = np.transpose(data, [2, 0, 1])
    # data = (data[0::2,:,:] + data[1::2,:,:]) / 2 
    wall_size = nlos_data['data_illumin_coordinates'][0,:].max() * 2
    data_theta = nlos_data['data_los_theta'].squeeze()
    deltaT = nlos_data['data_time_bin'][0][0] * 3e8

    return data, data_theta, deltaT, wall_size

def load_zaragoza_data(basedir):
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

    # data = (data[:,:,0::2] + data[:,:,1::2]) / 2 
    if 'deltaT' in nlos_data:
        deltaT = nlos_data['deltaT'][0][0]
    else:
        deltaT = 0.015/2

    if 'wall_size' in nlos_data:
        wall_size = nlos_data['wall_size'][0][0]
    else:
        wall_size = 0.636
    
    if 'highreso' in basedir:
        deltaT = 0.009/2
        wall_size = 0.4
    
    if 'S_sim_real' in basedir:
        deltaT = 0.015/2
    if ('bigbunny_cdt_32' or 'S_sim_real' or 'bigbunny_sim_32') in basedir:
        deltaT = 0.015/2
    # deltaT = nlos_data['deltaT'][0][0]
    # wall_size = nlos_data['wall_size'][0][0]
    # data = torch.from_numpy(data)

    return data, deltaT, wall_size

def load_cudaGL_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    # data = (data[:,:,0::2] + data[:,:,1::2]) / 2 
    # data = data[::4 ,::4, :]

    deltaT = nlos_data['deltaT'][0][0]
    wall_size = nlos_data['wall_size'][0][0]
    # data = torch.from_numpy(data)

    return data, deltaT, wall_size

def load_generated_gt(gtdir):
    volume_gt = scio.loadmat(gtdir)

    volume = volume_gt['Volume']
    xv = volume_gt['x'].reshape([-1])
    yv = volume_gt['y'].reshape([-1])
    zv = volume_gt['z'].reshape([-1])
    volume_vector = [xv,yv,zv]
    return volume, volume_vector

def load_data(dataset_type, datadir):
    if dataset_type == 'zaragoza':
        nlos_data, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT = load_zaragoza_data(datadir)
        Nz,Nx,Ny = nlos_data.shape
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])

        nlos_data = nlos_data / nlos_data.max() * 100
        wall_size = camera_grid_positions[0,0] - camera_grid_positions[0,-1]
        half_wall_size = wall_size / 2
        wall_resolution = int(np.sqrt(camera_grid_positions.shape[1]))

        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded zaragoza.')
        return nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny
        # return
    elif dataset_type == 'iqi':
        nlos_data, camera_grid_positions, deltaT, wall_size = load_iqi_data(datadir)
        Nz,Nx,Ny = 4096,64,64
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        # wall_size = camera_grid_positions[0,:].max() * 2
        half_wall_size = wall_size / 2
        nlos_data[nlos_data < 0] = 0
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded iqi.')
        return nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny
        # return
    elif dataset_type == 'iqi_los':
        data, data_theta, deltaT, wall_size  = load_iqi_los_data(datadir)
        Nz,Nx,Ny = 4096,64,64

        print(f'los_data: {data.shape}, data_theta: {data_theta.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded iqi los.')
        return data, data_theta, deltaT, wall_size ,Nz ,Nx ,Ny
        # return
    elif dataset_type == 'simtof':
        nlos_data, deltaT, wall_size = load_simtof_data(datadir)
        nlos_data = np.transpose(nlos_data, [2, 1, 0])
        Nz,Nx,Ny = nlos_data.shape
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        # nlos_data = nlos_data / nlos_data.max()
        nlos_data = nlos_data / nlos_data.max() * 100

        half_wall_size = wall_size / 2
        # wall_resolution = 8
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        # io.savemat('./test.mat',{'pos':camera_grid_positions})
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded simtof.')
        return nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny, 0, 0, 0, 0, 0
    
    elif dataset_type == 'cdt':
        nlos_data, deltaT, wall_size = load_simtof_data(datadir)
        nlos_data = np.transpose(nlos_data, [2, 1, 0])
        Nz,Nx,Ny = nlos_data.shape
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        # nlos_data = nlos_data / nlos_data.max()
        nlos_data = nlos_data / nlos_data.max() * 100

        c0 = 3e8
        n = 1.12
        c = c0/n
        mu_a = 0.53
        mu_s = 262
        ze = 0.0036
        zd = 0.0254

        half_wall_size = wall_size / 2
        # wall_resolution = 8
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded cdt.')
        # return nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny
        return nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny , c, mu_a, mu_s, ze, zd

    
    elif dataset_type == 'cudaGL':
        nlos_data, deltaT, wall_size = load_cudaGL_data(datadir)
        nlos_data = np.transpose(nlos_data, [2, 1, 0])
        Nz,Nx,Ny = nlos_data.shape
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        nlos_data = nlos_data / nlos_data.max() * 100

        half_wall_size = wall_size / 2
        # wall_resolution = 8
        # camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        # camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        # camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        # camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        # camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        # camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        # camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        # camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        # camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x, camera_grid_positions_z = np.meshgrid(camera_grid_positions_x, camera_grid_positions_z)
        camera_grid_positions_y = np.zeros_like(camera_grid_positions_x)
        camera_grid_positions_x = camera_grid_positions_x.reshape([1,-1])
        camera_grid_positions_y = camera_grid_positions_y.reshape([1,-1])
        camera_grid_positions_z = camera_grid_positions_z.reshape([1,-1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=0)
                
        # io.savemat('./test.mat',{'pos':camera_grid_positions})
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded cudaGL.')
        return nlos_data, camera_grid_positions, deltaT, wall_size, Nz ,Nx ,Ny        
    elif dataset_type == 'nlos_real':
        data = scio.loadmat(datadir)
        nlos_data = data['data']
        if 'parameters_wall_visible' in data:
            idx = data['parameters_wall_visible'].reshape(-1)
            idx = np.argwhere(idx==1).reshape(-1)
        # nlos_data = np.transpose(nlos_data, [2, 1, 0])
        Nz,Nx,Ny = nlos_data.shape

        wall_size = 1
        deltaT = data['deltaT'][0][0]
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        if 'parameters_wall_visible' in data:
            nlos_data = nlos_data[:,idx]
        nlos_data = nlos_data / nlos_data.max() * 100
        
        half_wall_size = wall_size / 2
        # wall_resolution = 8
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)
        if 'parameters_wall_visible' in data:
            camera_grid_positions = camera_grid_positions[:,idx]

        # io.savemat('./test.mat',{'pos':camera_grid_positions})
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded test')
        return nlos_data, camera_grid_positions, deltaT, wall_size, Nz ,Nx ,Ny , 0, 0, 0, 0, 0

    elif dataset_type == 'test':
        data = scio.loadmat(datadir)
        nlos_data = data['data']

        nlos_data = np.transpose(nlos_data, [2, 1, 0])
        Nz,Nx,Ny = nlos_data.shape

        wall_size = 0.5
        deltaT = 0.003
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        nlos_data = nlos_data / nlos_data.max() * 1000

        half_wall_size = wall_size / 2
        # wall_resolution = 8
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        # io.savemat('./test.mat',{'pos':camera_grid_positions})
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded test')
        # return
    else:
        raise NameError(f'Unknown dataset type: {dataset_type}.')

    