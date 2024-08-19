import os, sys
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

seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
hist2mse = lambda x, y : torch.mean((x - y) ** 2)

DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/', 
                        help='input data directory')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='nlos', 
                        help='options: nlos / genrated')

    # training options
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--N_iters", type=int, default=20, 
                        help='num of training iters')
 
    parser.add_argument("--use_encoding", action='store_true', 
                        help='use positional encoding or not')
    parser.add_argument("--use_normalization", action='store_true', 
                        help='use nomalization or not')

    parser.add_argument("--bin_batch", type=int, default=1, 
                        help='batch size (number of random bin per gradient step)')
    parser.add_argument("--hist_batch", type=int, default=4096, 
                        help='batch size (number of random hist per gradient step)')    

    parser.add_argument("--no_rho", action='store_true', 
                        help='use rho or not')
    parser.add_argument("--layer_nums", type=int, default=8, 
                        help='the number of layers in MLP')
    parser.add_argument("--hiddenlayer_dim", type=int, default=256, 
                        help='the dimmension of hidden layer')                    
    parser.add_argument("--encoding_dim", type=int, default=10, 
                        help='the dimmension of positional encoding, also L in the paper, attention that R is mapped to R^2L')
    parser.add_argument("--encoding_dim_view", type=int, default=10, 
                        help='the dimmension of positional encoding, also L in the paper, attention that R is mapped to R^2L')    

    parser.add_argument("--neglect_former_bins", action='store_true', 
                        help='when True, those former histogram bins will be neglected and not used in optimization. The threshold is computed automatically to ensure that neglected bins are zero')
    parser.add_argument("--neglect_former_nums", type=int, default=0, 
                        help='nums of former values ignored')
    parser.add_argument("--neglect_back_nums", type=int, default=0, 
                        help='nums of back values ignored')

    parser.add_argument("--test_accurate_sampling", action='store_true', 
                        help='when True, the sampling function will sample from the known object box area, rather than the whole function')
    parser.add_argument("--sampling_points_nums", type=int, default=16, 
                        help='number of sampling points in one direction, so the number of all sampling points is the square of this value')
                 
    return parser


def render():

    parser = config_parser()
    args = parser.parse_args()
    writer = SummaryWriter(args.basedir + args.expname)

    # Load data
    if args.dataset_type == 'nlos':
        cdt_data, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT = load_cdt_data(args.datadir)
        # cdt_data = np.transpose(cdt_data, [1, 2, 0])
        # print(cdt_data.max())
        cdt_data = cdt_data / cdt_data.max() * 100
        wall_size = 1
        magic_number = 0.5
        # print(cdt_data.max())
        # cdt_data = cdt_data * 100
        # cdt_data = cdt_data / cdt_data.max()
        print(cdt_data.shape, camera_grid_positions.shape)
        print('Loaded nlos')
        # return
    elif args.dataset_type == 'iqi':
        cdt_data, camera_grid_positions, deltaT = load_iqi_data(args.datadir)
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        wall_size = camera_grid_positions[0,:].max() * 2
        magic_number = wall_size / 2
        print(cdt_data.shape, camera_grid_positions.shape, deltaT, wall_size)
        print('Loaded iqi')
        # return
    elif args.dataset_type == 'generated':
        cdt_data = load_generated_data(args.datadir)
        cdt_data = np.transpose(cdt_data, [2, 0, 1])
        # cdt_data = cdt_data / cdt_data.max() * 100
        # print(cdt_data.max())
        deltaT = 0.0012
        wall_size = 0.55
        magic_number = wall_size / 2
        wall_resolution = 64
        camera_grid_positions_z = np.linspace(-magic_number, magic_number, wall_resolution)
        camera_grid_positions_z = np.outer(camera_grid_positions_z, np.ones_like(camera_grid_positions_z))
        camera_grid_positions_y = np.zeros([wall_resolution*wall_resolution,1])
        camera_grid_positions_x = np.linspace(-magic_number, magic_number, wall_resolution)
        camera_grid_positions_x = np.outer(np.ones_like(camera_grid_positions_x), camera_grid_positions_x)
        camera_grid_positions_z = camera_grid_positions_z.flatten().reshape([-1,1])
        camera_grid_positions_x = camera_grid_positions_x.flatten().reshape([-1,1])
        camera_grid_positions = np.concatenate((camera_grid_positions_x, camera_grid_positions_y, camera_grid_positions_z), axis=1)
        camera_grid_positions = camera_grid_positions.swapaxes(0,1)

        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]
        print(cdt_data.shape, camera_grid_positions.shape)
        print('Loaded nlos')
        # return
    elif args.dataset_type == 'real':
        volume_position = [0 , 1.08, 0]
        volume_size = [0.5]

        cdt_data = io.loadmat(args.datadir)

        data = cdt_data['data']
        data = data.reshape([64, 64, 4096])
        temp = cdt_data['positions']
        camera_grid_positions = np.zeros([3, 4096])
        camera_grid_positions[0,:] = temp[0,:]
        camera_grid_positions[2,:] = temp[1,:]
        deltaT = 0.0012 / 2
        cdt_data = np.transpose(data,(2, 0, 1))
        wall_size = 0.8
        print('Loaded real data')    
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

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
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(histogram_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)


    # Construct our model
    if args.use_encoding:
        model = Network(D = args.layer_nums, H = args.hiddenlayer_dim, input_ch = 6 * args.encoding_dim, input_ch_views = 4 * args.encoding_dim_view, skips=[4], no_rho=args.no_rho)
    else:
        model = Network(D = args.layer_nums, H = args.hiddenlayer_dim, input_ch = 3, input_ch_views = 2,  skips=[4], no_rho=args.no_rho)

    # Construct our loss function and an Optimizer.
    # model = torch.nn.DataParallel(model)
    model = torch.load(model_path + 'epoch20_3082.pt')
    # model.eval()
    model = model.to(device)

    # ignore some useless bins
    if args.neglect_former_bins:
        data_start = args.neglect_former_nums
        data_end = cdt_data.shape[0] - args.neglect_back_nums
        cdt_data = cdt_data
        print('all bins < ', data_start, ' and bins >', data_end, ' are neglected')
        # return
    else:
        data_start = 0
        data_end = cdt_data.shape[2]

    # Pre-process
    pmin = torch.Tensor([-wall_size/2 - data_end * deltaT, - data_end * deltaT, -wall_size/2 - data_end * deltaT, 0, -np.pi]).float().to(device)
    pmax = torch.Tensor([wall_size/2 + data_end * deltaT, - data_start * deltaT, wall_size/2 + data_end * deltaT, np.pi, 0]).float().to(device)
    
    cdt_data = torch.Tensor(cdt_data).to(device)
    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)

    reso = 64
    input_x, input_y, input_z = torch.meshgrid(
        torch.linspace(-magic_number, magic_number, reso),
        torch.linspace(data_start * deltaT, data_end * deltaT, reso) * -1,
        torch.linspace(-magic_number, magic_number, reso)
    )

    input_x = input_x.reshape([-1, 1])
    input_y = input_y.reshape([-1, 1])
    input_z = input_z.reshape([-1, 1])

    test_input = torch.cat((input_x, input_y, input_z, torch.ones_like(input_x) * np.pi/2, torch.ones_like(input_x) * np.pi/2 * -1), axis = 1)
    test_input = (test_input - pmin) / (pmax - pmin)
    test_input_pe = encoding_sph_tensor(test_input.to(device), args.encoding_dim, args.encoding_dim_view,  args.no_rho, device)

    with torch.no_grad():
        output = model(test_input_pe)
        temp = (output[:,0] * output[:,1]).reshape([reso, reso, reso])
        # temp_img = temp.max(axis = 1).values
        # temp_img = temp.max(axis = 0).values
        temp_img = temp.max(axis = 2).values
        plt.imshow(temp_img.cpu().data.numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.savefig(img_path + 'result_top')
        plt.close()

if __name__=='__main__':
    # python run_netf.py --config configs/nlos.txt
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    render()
