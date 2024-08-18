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

    cdt_data = scio.loadmat(basedir)

    data = cdt_data['data_transients'][:,:4096]
    data = np.transpose(data, [1, 0])
    illumination_coords = cdt_data['data_illumin_coordinates']
    illumination_coords_x = illumination_coords[0,:].reshape([1,-1])
    illumination_coords_y = np.zeros([1, illumination_coords.shape[1]])
    illumination_coords_z = illumination_coords[1,:].reshape([1,-1])
    camera_grid_positions = np.concatenate([illumination_coords_x, illumination_coords_y, illumination_coords_z], axis=0)

    deltaT = cdt_data['data_time_bin'][0][0] * 3e8
    wall_size = camera_grid_positions[0,:].max() * 2

    mask = cdt_data['parameters_wall_visible']
    mask = (mask.squeeze() == 1)
    # print(mask.shape)
    # print(data.shape, camera_grid_positions.shape)
    # print(data[:, mask].shape, camera_grid_positions[:, mask].shape)
    data = data[:, mask]
    camera_grid_positions = camera_grid_positions[:, mask]
    return data, camera_grid_positions, deltaT, wall_size

def load_iqi_los_data(basedir):

    cdt_data = scio.loadmat(basedir)

    data = cdt_data['data_los_transients'][:,:4096]
    data = np.transpose(data, [1, 0])
    wall_size = cdt_data['data_illumin_coordinates'][0,:].max() * 2
    data_theta = cdt_data['data_los_theta'].squeeze()
    deltaT = cdt_data['data_time_bin'][0][0] * 3e8

    return data, data_theta, deltaT, wall_size

def load_zaragoza_data(basedir):
    # cdt_data = h5py.File(basedir, 'r')
    cdt_data = scio.loadmat(basedir)

    data = cdt_data['data']
    data = data[:, :, :]
    camera_grid_positions = cdt_data['cameraGridPositions']
    camera_grid_points = cdt_data['cameraGridPoints'][0,:]
    volume_position = cdt_data['hiddenVolumePosition']
    volume_size = cdt_data['hiddenVolumeSize']
    deltaT = cdt_data['deltaT'][0,:][0]

    return data, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT

def load_real_data(basedir):
    cdt_data = scio.loadmat(basedir)

    data = cdt_data['data']
    data = data[:, :, :]
    wall_size = cdt_data['wall_size'][0,:][0]
    deltaT = cdt_data['time_bin'][0,:][0] * cdt_data['speed_light'][0,:][0]
    return data, wall_size, deltaT

def load_generated_data(basedir):
    cdt_data = scio.loadmat(basedir)
    data = cdt_data['data']
    data = data[:, :, :]
    return data

def load_simtof_data(basedir):
    cdt_data = scio.loadmat(basedir)

    data = cdt_data['data']
    data = data[:, :, :]

    if 'deltaT' in cdt_data:
        deltaT = cdt_data['deltaT'][0][0]
    else:
        deltaT = 0.015/2

    if 'wall_size' in cdt_data:
        wall_size = cdt_data['wall_size'][0][0]
    else:
        wall_size = 0.636
    
    if 'highreso' in basedir:
        deltaT = 0.009/2
        wall_size = 0.4
    
    if 'S_sim_real' in basedir:
        deltaT = 0.015/2
    if ('bigbunny_cdt_32' or 'S_sim_real' or 'bigbunny_sim_32') in basedir:
        deltaT = 0.015/2

    return data, deltaT, wall_size

def load_cudaGL_data(basedir):
    cdt_data = scio.loadmat(basedir)
    data = cdt_data['data']
    data = data[:, :, :]
    deltaT = cdt_data['deltaT'][0][0]
    wall_size = cdt_data['wall_size'][0][0]

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
        cdt_data, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT = load_zaragoza_data(datadir)
        Nz,Nx,Ny = cdt_data.shape
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])

        cdt_data = cdt_data / cdt_data.max() * 100
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

        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded zaragoza.')
        return cdt_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny
        # return
    elif dataset_type == 'iqi':
        cdt_data, camera_grid_positions, deltaT, wall_size = load_iqi_data(datadir)
        Nz,Nx,Ny = 4096,64,64
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]

        half_wall_size = wall_size / 2
        cdt_data[cdt_data < 0] = 0
        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded iqi.')
        return cdt_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny
        # return
    elif dataset_type == 'iqi_los':
        data, data_theta, deltaT, wall_size  = load_iqi_los_data(datadir)
        Nz,Nx,Ny = 4096,64,64

        print(f'los_data: {data.shape}, data_theta: {data_theta.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded iqi los.')
        return data, data_theta, deltaT, wall_size ,Nz ,Nx ,Ny
        # return
    elif dataset_type == 'simtof':
        cdt_data, deltaT, wall_size = load_simtof_data(datadir)
        cdt_data = np.transpose(cdt_data, [2, 1, 0])
        Nz,Nx,Ny = cdt_data.shape
        wall_resolution = cdt_data.shape[1]
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])
        cdt_data = cdt_data / cdt_data.max() * 100

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

        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded simtof.')
        return cdt_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny, 0, 0, 0, 0, 0
    
    elif dataset_type == 'cdt':
        data = scio.loadmat(datadir)
        cdt_data = data['data']

        deltaT = 0.015/2
        wall_size = 0.636
        mu_a = 0.53
        mu_s = 262
        n = 1.12
        zd = 0.0254 

        if 'deltaT' in data:
            deltaT = data['deltaT'][0][0]
        if 'wall_size' in data:
            wall_size = data['wall_size'][0][0]
        if 'ua' in data:
            mu_a = data['ua'][0][0]
        if 'us' in data:
            mu_s = data['us'][0][0]
        if 'n' in data:
            n = data['n'][0][0]
        if 'd' in data:
            zd = data['d'][0][0]

        cdt_data = np.transpose(cdt_data, [2, 1, 0])
        Nz,Nx,Ny = cdt_data.shape
        wall_resolution = cdt_data.shape[1]
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])
        cdt_data = cdt_data / cdt_data.max() * 100

        c0 = 3e8
        c = c0/n
        ze = 0
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

        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded cdt.')
        return cdt_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny , c, mu_a, mu_s, n, zd

    
    elif dataset_type == 'cudaGL':
        cdt_data, deltaT, wall_size = load_cudaGL_data(datadir)
        cdt_data = np.transpose(cdt_data, [2, 1, 0])
        Nz,Nx,Ny = cdt_data.shape
        wall_resolution = cdt_data.shape[1]
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])
        cdt_data = cdt_data / cdt_data.max() * 100

        half_wall_size = wall_size / 2

        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x, camera_grid_positions_z = np.meshgrid(camera_grid_positions_x, camera_grid_positions_z)
        camera_grid_positions_y = np.zeros_like(camera_grid_positions_x)
        camera_grid_positions_x = camera_grid_positions_x.reshape([1,-1])
        camera_grid_positions_y = camera_grid_positions_y.reshape([1,-1])
        camera_grid_positions_z = camera_grid_positions_z.reshape([1,-1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=0)
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded cudaGL.')
        return cdt_data, camera_grid_positions, deltaT, wall_size, Nz ,Nx ,Ny      
        
    elif dataset_type == 'nlos_real':
        data = scio.loadmat(datadir)
        cdt_data = data['data']
        if 'parameters_wall_visible' in data:
            idx = data['parameters_wall_visible'].reshape(-1)
            idx = np.argwhere(idx==1).reshape(-1)

        Nz,Nx,Ny = cdt_data.shape
        wall_size = 1
        deltaT = data['deltaT'][0][0]
        wall_resolution = cdt_data.shape[1]
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])
        if 'parameters_wall_visible' in data:
            cdt_data = cdt_data[:,idx]
        cdt_data = cdt_data / cdt_data.max() * 100
        
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

        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded test')
        return cdt_data, camera_grid_positions, deltaT, wall_size, Nz ,Nx ,Ny , 0, 0, 0, 0, 0

    elif dataset_type == 'test':
        data = scio.loadmat(datadir)
        cdt_data = data['data']

        cdt_data = np.transpose(cdt_data, [2, 1, 0])
        Nz,Nx,Ny = cdt_data.shape

        wall_size = 0.5
        deltaT = 0.003
        wall_resolution = cdt_data.shape[1]
        cdt_data = cdt_data.reshape([cdt_data.shape[0], -1])
        cdt_data = cdt_data / cdt_data.max() * 1000

        half_wall_size = wall_size / 2
        camera_grid_positions_z = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-half_wall_size, half_wall_size, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'cdt_data: {cdt_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded test')
        # return
    else:
        raise NameError(f'Unknown dataset type: {dataset_type}.')

    