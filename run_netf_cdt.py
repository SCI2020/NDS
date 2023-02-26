import os, sys
from tkinter import N
import numpy as np
# import json
# import random
import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_netf_helpers import *
from MLP import *
from utils.config_parser import config_parser

from utils.load_nlos import *
from scipy import io

import trimesh
import mcubes

seed = 3407
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# DEBUG = False

def train():

    parser = config_parser()
    args = parser.parse_args()
    writer = SummaryWriter(args.basedir + args.expname)

    print('-------------------' + args.basedir + args.expname + '----------------------')
    ################################################################################
    # Load data
    # c, mu_a, mu_s, ze, wall_size, zmax, zd
    nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny , c, mu_a, mu_s, ze, zd = load_data(args.dataset_type, args.datadir)
    # nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny = load_data(args.dataset_type, args.datadir)

    volume_size = np.array([wall_size/2]),np.array([deltaT*Nz/2]),np.array([wall_size/2])
    volume_position = [0 , deltaT*Nz/2, 0]

    # return

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
    obj_path = os.path.join(basedir, expname, 'obj/')
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(histogram_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(obj_path, exist_ok=True)

    ################################################################################
    # Construct our model
    if args.tcnn:
        model = NGPNetwork_tcnn(
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
    else:
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
    # criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, betas=(0.9, 0.99), eps=1e-15)
    optimizer.zero_grad()

    ################################################################################
    # ignore some useless bins
    data_start = args.neglect_former_nums
    # data_end = nlos_data.shape[0] - args.neglect_back_nums
    data_end = args.neglect_back_nums
    nlos_data = nlos_data[data_start:data_end,:]
    nlos_data = torch.Tensor(nlos_data).to(device)
    print(f'All bins < {data_start} and bins > {data_end} are neglected. Ignored data: {nlos_data.shape}')

    # Pre-process
    pmin = torch.Tensor([-(wall_size/2) - data_end * deltaT, -1e-7, -(wall_size/2) - data_end * deltaT]).float().to(device)
    pmax = torch.Tensor([wall_size/2 + data_end * deltaT, data_end * deltaT, wall_size/2 + data_end * deltaT]).float().to(device)
    # pmin = torch.Tensor([-wall_size/2, -1e-7, -wall_size/2]).float().to(device)
    # pmax = torch.Tensor([wall_size/2, data_end * deltaT, wall_size/2]).float().to(device)    
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
    # bin_batch = args.bin_batch
    lr_decay_rate = args.lr_decay_rate
    i_loss = args.i_loss
    i_hist = args.i_hist
    i_image = args.i_image
    i_model = args.i_model
    i_print = args.i_print
    i_obj = args.i_obj    
    ################################################################################
    # Prepare log points and normalize the coords to [-1, 1]
    # reso = args.reso
    # reso = 256
    reso = 64
    input_x, input_y, input_z = torch.meshgrid(
        torch.linspace(-(wall_size / 2), (wall_size / 2), reso),
        torch.linspace(0, Nz * deltaT, reso),
        # torch.linspace(data_start * deltaT, data_end * deltaT, reso),
        torch.linspace(-(wall_size / 2), (wall_size / 2), reso)
    )
    input_x = input_x.reshape([-1, 1])
    input_y = input_y.reshape([-1, 1])
    input_z = input_z.reshape([-1, 1])
    test_input_coord = torch.cat((input_x, input_y, input_z), axis = 1)
    test_input_dir = torch.cat((torch.zeros_like(input_x), torch.ones_like(input_x), torch.zeros_like(input_x)), axis = 1)
    test_input_coord = (test_input_coord - pmin) / (pmax - pmin) * 2 - 1
    # test_input_coord = (test_input_coord - pmin) / (pmax - pmin)
    # test_input_dir = (test_input_dir - pmin_dir) / (pmax_dir - pmin_dir)
    
    if args.shift:
        psf = set_cdt_completekernel_torch(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, Nz*deltaT*2, zd, device, n_dipoles = 20)
    else:
        psf = set_cdt_completekernel_noshift(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, Nz*deltaT*2, zd, device, n_dipoles = 7)
    # pdb.set_trace()

    print('------------------------------------------')
    print('Training Begin')

    global_step = 0
    loss_global = []
    time0 = time.time()
    for i in trange(0, N_iters):
        ################################################################################
        # random sampling and normalize
        # pdb.set_trace()

        bin_batch = nlos_data.shape[0]
        index_rand = torch.arange(0, nlos_data.shape[0])
        # index_rand = torch.randint(0, nlos_data.shape[0], (bin_batch,))
    
        r_ = r + torch.rand(r.shape) * deltaT
        input_coord, input_dir, theta, _, _, _, r_batch, cut_index = spherical_sample_bin_tensor_bbox(camera_grid_positions[:,index_rand], r_[index_rand].squeeze(), args.sampling_points_nums, volume_position, volume_size)
        # input_coord, input_dir, theta, _, _, _, r_batch = spherical_sample_bin_tensor(camera_grid_positions[:,index_rand], r_[index_rand].squeeze(), args.sampling_points_nums)
        sin_theta = torch.sin(theta)
        input_coord = (input_coord - pmin) / (pmax - pmin) * 2 - 1

        # predict transient
        # pdb.set_trace()
        sigma, color = model(input_coord, input_dir)    
        # print(f"Barrrrrrrrrr: {sigma.dtype}")
        network_res = torch.mul(sigma, color)
        network_res = torch.mul(network_res, sin_theta)
        if args.bbox:
            network_res = network_res.squeeze() * cut_index

        network_res = network_res.reshape(bin_batch, args.sampling_points_nums*args.sampling_points_nums)
        if args.occlusion: 
            sigma = sigma.reshape(data_end-data_start, Nx, Ny, args.sampling_points_nums*args.sampling_points_nums)
            occlusion = torch.cumsum(sigma, axis = 0)
            occlusion = torch.exp(- occlusion)
            occlusion = occlusion.reshape(bin_batch, args.sampling_points_nums*args.sampling_points_nums)
            network_res = torch.mul(network_res, occlusion)
    
        network_res = torch.sum(network_res, 1)
        network_res = network_res / (r_batch ** 2)

        nlos_pad = torch.zeros([Nz,Nx,Ny])
        nlos_pad[data_start:data_end,:,:] = network_res.reshape([data_end-data_start, Nx, Ny])

        if args.shift:
            cdt_conv = torch.fft.ifftn(torch.mul(torch.fft.fftn(nlos_pad, s=(Nz*2, Nx*2-1, Ny*2-1)), psf)).real
        else:
            cdt_conv = torch.fft.ifftn(torch.mul(torch.fft.fftn(nlos_pad, s=(Nz*2, Nx*2, Ny*2)), psf)).real
        predict_cdt = cdt_conv.squeeze()[:Nz,:Nx,:Ny]
        predict_cdt = predict_cdt[data_start:data_end,:Nx,:Ny].reshape([-1,1])[index_rand].squeeze()

        nlos_histogram = nlos_data[index_rand].squeeze()
        # pre_histogram = network_res[nlos_histogram>=0]
        pre_histogram = predict_cdt[nlos_histogram>=0]
        nlos_histogram = nlos_histogram[nlos_histogram>=0]


        # update
        # print(f"Fooooooooo: {pre_histogram.dtype, nlos_histogram.dtype}")
        loss = criterion(pre_histogram, nlos_histogram)
        # loss = criterion(pre_histogram.double(), nlos_histogram.double())
        # loss = criterion_l1(network_res, nlos_histogram)
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
            plt.plot(pre_histogram.cpu().detach().numpy(), alpha = 0.5, label='predicted')
            plt.title('Histogram_iter' + str(i+1))
            plt.legend(loc='upper right')
            plt.savefig(histogram_path + 'histogram_' + str(i+1))
            plt.close()

        # log recon result
        if (i+1) % i_image == 0:
            with torch.no_grad():
                temp_sigma, temp_color = model(test_input_coord, test_input_dir)
                temp = (temp_sigma * temp_color).reshape([reso, reso, reso])

                # pdb.set_trace()
                temp[:,-10:,:] = 0
                temp[:,:data_start*reso//Nz,:] = 0
                temp[:,data_end*reso//Nz:,:] = 0

                temp_img = temp.max(axis = 1).values
                # plt.imshow(temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                # plt.axis('off')
                # plt.savefig(img_path + 'result_' + str(i+1) + '_XOY')
                plt.imsave(img_path + 'result_' + str(i+1) + '_XOY.png', temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.close()

                temp_img = temp.max(axis = 0).values
                # plt.imshow(temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                # plt.axis('off')
                # plt.savefig(img_path + 'result_' + str(i+1) + '_Y0Z')
                plt.imsave(img_path + 'result_' + str(i+1) + '_YOZ.png', temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.close()

                temp_img = temp.max(axis = 2).values
                # plt.imshow(temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                # plt.axis('off')
                # plt.savefig(img_path + 'result_' + str(i+1) + '_X0Z')
                plt.imsave(img_path + 'result_' + str(i+1) + '_XOZ.png', temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.close()
                # io.savemat(result_path + 'vol_' + str(i+1) + '.mat' , {'res_vol': temp.cpu().data.numpy().squeeze()})

        # log recon obj
        if (i+1) % i_obj == 0:
            with torch.no_grad():
                temp_sigma, temp_color = model(test_input_coord, test_input_dir)
                temp = (temp_sigma * temp_color).reshape([reso, reso, reso]).cpu().data.numpy().squeeze()
                threshold = args.obj_threshold
                vertices, triangles = mcubes.marching_cubes(temp, threshold * temp.max())
                mesh = trimesh.Trimesh(vertices, triangles)
                trimesh.repair.fill_holes(mesh)
                # mesh.show()
                mesh.export(obj_path + 'obj_' + str(i+1) + '.obj')
    
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
