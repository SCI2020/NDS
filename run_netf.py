import os, sys
from tkinter import N
import numpy as np
# import json
# import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_netf_helpers import *
from MLP import *

from load_nlos import *
from scipy import io

seed = 3407
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')

    # exp options
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')

    # dataset options
    parser.add_argument("--datadir", type=str, default='./data/', 
                        help='input data directory')
    parser.add_argument("--dataset_type", type=str, default='nlos', 
                        help='options: nlos / genrated')
    parser.add_argument("--neglect_zero_bins", action='store_true', 
                        help='when True, those zero histogram bins will be neglected and not used in optimization. The threshold is computed automatically to ensure that neglected bins are zero')
    parser.add_argument("--neglect_former_nums", type=int, default=0, 
                        help='nums of former values ignored')
    parser.add_argument("--neglect_back_nums", type=int, default=0, 
                        help='nums of back values ignored')

    # MLP options
    parser.add_argument("--encoding", type=str, default='hashgrid', 
                        help='encoding type for position')
    parser.add_argument("--encoding_dir", type=str, default='sphere_harmonics', 
                        help='encoding type for direction')
    parser.add_argument("--num_layers", type=int, default=2, 
                        help='the number of layers for sigma')
    parser.add_argument("--hidden_dim", type=int, default=64, 
                        help='the dimmension of hidden layer for sigma net')                
    parser.add_argument("--geo_feat_dim", type=int, default=15, 
                        help='the dimmension of geometric feature')
    parser.add_argument("--num_layers_color", type=int, default=3, 
                        help='the number of layers for color')   
    parser.add_argument("--hidden_dim_color", type=int, default=64, 
                        help='the dimmension of hidden layer for color net')                         
    parser.add_argument("--bound", type=int, default=1,
                        help='boundry of the scene')   
    parser.add_argument("--reso", type=int, default=64,
                        help='the result resolution')  
    # training options
    parser.add_argument("--N_iters", type=int, default=20, 
                        help='num of training iters')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, 
                        help='learning rate decay rate')
    parser.add_argument("--bin_batch", type=int, default=2048, 
                        help='batch size (number of random bin per gradient step)')
    parser.add_argument("--sampling_points_nums", type=int, default=16, 
                        help='number of sampling points in one direction, so the number of all sampling points is the square of this value')

    # log options 
    parser.add_argument("--i_loss", type=int, default=100, 
                        help='num of iters to log loss') 
    parser.add_argument("--i_hist", type=int, default=100, 
                        help='num of iters to log histogram')  
    parser.add_argument("--i_image", type=int, default=100, 
                        help='num of iters to log result image') 
    parser.add_argument("--i_model", type=int, default=1000, 
                        help='num of iters to log model') 
    parser.add_argument("--i_print", type=int, default=100, 
                        help='num of iters to log print') 

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    writer = SummaryWriter(args.basedir + args.expname)

    print('------------------------------------------')
    ################################################################################
    # Load data
    if args.dataset_type == 'nlos':
        nlos_data, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT = load_nlos_data(args.datadir)
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        nlos_data = nlos_data / nlos_data.max()
        # nlos_data = np.transpose(nlos_data, [1, 2, 0])
        # print(nlos_data.max())
        nlos_data = nlos_data / nlos_data.max() * 100
        wall_size = 1
        magic_number = wall_size / 2

        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded nlos')
        # return
    elif args.dataset_type == 'iqi':
        nlos_data, camera_grid_positions, deltaT = load_iqi_data(args.datadir)
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        nlos_data = nlos_data / nlos_data.max() * 100
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        wall_size = camera_grid_positions[0,:].max() * 2
        magic_number = wall_size / 2
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded iqi')
        # return
    elif args.dataset_type == 'generated':
        nlos_data = load_generated_data(args.datadir)
        # nlos_data = np.transpose(nlos_data, [2, 0, 1])
        nlos_data = np.transpose(nlos_data, [2, 1, 0])
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        nlos_data = nlos_data / nlos_data.max()
        deltaT = 0.0012
        wall_size = 1
        magic_number = wall_size / 2
        # wall_resolution = 8
        camera_grid_positions_z = np.linspace(-magic_number, magic_number, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-magic_number, magic_number, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        # io.savemat('./test.mat',{'pos':camera_grid_positions})
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded nlos')
        # return
    elif args.dataset_type == 'simtof':
        nlos_data, deltaT = load_simtof_data(args.datadir)
        # nlos_data = np.transpose(nlos_data, [2, 0, 1])
        nlos_data = np.transpose(nlos_data, [2, 1, 0])
        wall_resolution = nlos_data.shape[1]
        nlos_data = nlos_data.reshape([nlos_data.shape[0], -1])
        nlos_data = nlos_data / nlos_data.max() * 100
        # deltaT = 0.0012
        wall_size = 1
        magic_number = wall_size / 2
        # wall_resolution = 8
        camera_grid_positions_z = np.linspace(-magic_number, magic_number, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-magic_number, magic_number, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        # io.savemat('./test.mat',{'pos':camera_grid_positions})
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded simtof')
    elif args.dataset_type == 'real':
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]

        nlos_data = io.loadmat(args.datadir)

        data = nlos_data['data']
        data = data.reshape([64, 64, 4096])
        temp = nlos_data['positions']
        camera_grid_positions = np.zeros([3, 4096])
        camera_grid_positions[0,:] = temp[0,:]
        camera_grid_positions[2,:] = temp[1,:]
        deltaT = 0.0012 / 2
        nlos_data = np.transpose(data,(2, 0, 1))
        wall_size = 0.8
        print(f'nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, deltaT: {deltaT}, wall_size: {wall_size}.')
        print('Loaded real data')    
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    ################################################################################
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    result_path = os.path.join(basedir, expname, 'result/')
    model_path = os.path.join(basedir, expname, 'model/')
    histogram_path = os.path.join(basedir, expname, 'histogram/')
    img_path = os.path.join(basedir, expname, 'image/')
    loss_path = os.path.join(basedir, expname, 'loss/')
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(histogram_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)

    # Construct our model
    model = NGPNetwork(
        encoding = args.encoding,
        encoding_dir = args.encoding_dir,
        num_layers = args.num_layers,
        hidden_dim = args.hidden_dim,
        geo_feat_dim = args.geo_feat_dim,
        num_layers_color = args.num_layers_color,
        hidden_dim_color = args.hidden_dim_color,
        bound = args.bound,
        reso = args.reso
    )
    # print(model)
    # return   
    
    ################################################################################
    # Construct our loss function and an Optimizer.
    model = torch.nn.DataParallel(model)
    # model = torch.load("./model/pre_train.pt")
    model = model.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')
    criterion_l1 = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, betas=(0.9, 0.99), eps=1e-15)
    optimizer.zero_grad()

    ################################################################################
    # ignore some useless bins
    data_start = args.neglect_former_nums
    data_end = nlos_data.shape[0] - args.neglect_back_nums
    nlos_data = nlos_data[data_start:data_end,:]
    nlos_data = torch.Tensor(nlos_data).to(device)
    print(f'All bins < {data_start} and bins > {data_end} are neglected. Ignored data: {nlos_data.shape}')

    # Pre-process
    pmin = torch.Tensor([-wall_size/2 - data_end * deltaT, -1e-7, -wall_size/2 - data_end * deltaT]).float().to(device)
    pmax = torch.Tensor([wall_size/2 + data_end * deltaT, data_end * deltaT, wall_size/2 + data_end * deltaT]).float().to(device)
    pmin_dir = torch.Tensor([-1, -1, -1]).float().to(device)
    pmax_dir = torch.Tensor([1, 1, 1]).float().to(device)

    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
    camera_grid_positions = camera_grid_positions.reshape([3,1,-1]).expand(-1, data_end-data_start, -1)

    r_min = data_start * deltaT
    r_max = data_end * deltaT
    num_r = data_end - data_start
    r = torch.linspace(r_min, r_max , num_r)
    r = r.reshape([-1,1]).expand(-1, nlos_data.shape[1])

    camera_grid_positions = camera_grid_positions.reshape([3,-1])
    nlos_data = nlos_data.reshape([-1,1])
    r = r.reshape([-1,1])
    print(f'Pre-precoess done. nlos_data: {nlos_data.shape}, camera_grid_positions: {camera_grid_positions.shape}, r: {r.shape}')

    N_iters = args.N_iters
    bin_batch = args.bin_batch
    lr_decay_rate = args.lr_decay_rate
    i_loss = args.i_loss
    i_hist = args.i_hist
    i_image = args.i_image
    i_model = args.i_model
    i_print = args.i_print
    ################################################################################
    # Prepare log points and normalize the coords to [-1, 1]
    reso = args.reso
    input_x, input_y, input_z = torch.meshgrid(
        torch.linspace(-magic_number, magic_number, reso),
        torch.linspace(data_start * deltaT, data_end * deltaT, reso),
        torch.linspace(-magic_number, magic_number, reso)
    )
    input_x = input_x.reshape([-1, 1])
    input_y = input_y.reshape([-1, 1])
    input_z = input_z.reshape([-1, 1])
    test_input_coord = torch.cat((input_x, input_y, input_z), axis = 1)
    test_input_dir = torch.cat((torch.zeros_like(input_x), torch.ones_like(input_x), torch.zeros_like(input_x)), axis = 1)
    test_input_coord = (test_input_coord - pmin) / (pmax - pmin) * 2 - 1


    print('------------------------------------------')
    print('Training Begin')

    global_step = 0
    loss_global = []
    time0 = time.time()
    for i in trange(0, N_iters):
        ################################################################################
        # random sampling and normalize
        # print('------------------------------------------')
        # time0 = time.time()
        index_rand = torch.randint(0, nlos_data.shape[0], (bin_batch,))
        # print(f"Index time: {time.time() - time0}")
        # time0 = time.time()
        input_coord, input_dir, theta, _, _, _, r_batch = spherical_sample_bin_tensor(camera_grid_positions[:,index_rand], r[index_rand].squeeze(), args.sampling_points_nums)
        # print(f"Sampling time: {time.time() - time0}")
        # time0 = time.time()
        sin_theta = torch.sin(theta)
        input_coord = (input_coord - pmin) / (pmax - pmin) * 2 - 1
        # r_batch_coe = r_batch.pow(2) / r_batch.pow(2)
        # print(r_batch.shape, r_batch.max(), r_batch.min())
        # return
        # predict transient
        # print(f"Time: {time.time() - time0}")
        # time0 = time.time()
        sigma, color = model(input_coord, input_dir)
        # print(f"Model time: {time.time() - time0}")
        # time0 = time.time()        
        network_res = torch.mul(sigma, color)
        network_res = torch.mul(network_res, sin_theta)
        network_res = network_res.reshape(bin_batch, args.sampling_points_nums*args.sampling_points_nums)
        network_res = torch.sum(network_res, 1)
        # print(network_res.shape, r_batch.shape, (r_batch ** 2).shape)
        # return
        network_res = network_res / (r_batch ** 2)
        # print(f"Cal time: {time.time() - time0}")
        # time0 = time.time() 
        nlos_histogram = nlos_data[index_rand].squeeze()

        # update
        # loss = criterion(network_res, nlos_histogram) + 20 * criterion_l1(sigma, torch.zeros_like(sigma))
        loss = criterion(network_res, nlos_histogram)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # lr schedule
        decay_rate = lr_decay_rate
        decay_steps = N_iters
        global_step += 1
        if global_step <= decay_steps:
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

        ################################################################################
        writer.add_scalar('loss', loss.item(), global_step=(i+1))  
        loss_global.append(loss.item())

        # log loss
        if (i+1) % i_loss == 0:
            plt.plot(loss_global)
            plt.title('Loss')
            plt.savefig(loss_path + 'Loss')
            plt.close()

        # log histogram
        if (i+1) % i_hist == 0:
            plt.plot(nlos_histogram.cpu(), alpha = 0.5, label = 'data')
            plt.plot(network_res.cpu().detach().numpy(), alpha = 0.5, label='predicted')
            plt.title('Histogram_iter' + str(i+1))
            plt.legend(loc='upper right')
            plt.savefig(histogram_path + 'histogram_' + str(i+1))
            plt.close()

        # log recon result
        if (i+1) % i_image == 0:
            with torch.no_grad():
                temp_sigma, temp_color = model(test_input_coord, test_input_dir)
                temp = (temp_sigma * temp_color).reshape([reso, reso, reso])
                temp_img = temp.max(axis = 1).values
                plt.imshow(temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.axis('off')
                plt.savefig(img_path + 'result_' + str(i+1) + '_XOY')
                plt.close()
                temp_img = temp.max(axis = 0).values
                plt.imshow(temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.axis('off')
                plt.savefig(img_path + 'result_' + str(i+1) + '_Y0Z')
                plt.close()

                io.savemat(result_path + 'vol_' + str(i+1) + '.mat' , {'res_vol': temp.cpu().data.numpy().squeeze()})
        
        # log model
        if (i+1) % i_model == 0:
            model_name = model_path + 'model_' + str(i+1) + '.pt'
            torch.save(model, model_name)

        # log print
        if (i+1) % i_print == 0: 
            dt = time.time()-time0
            tqdm.write(f"[TRAIN] Iter: {i+1} / {N_iters}, Loss: {loss.item()}, lrate: {new_lrate}, TIME: {dt}")
            time0 = time.time()

if __name__=='__main__':
    # CUDA_VISIBLE_DEVICES=x python run_netf.py --config configs/X/X.txt
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
