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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    device1 = torch.device(f"cuda:{5%torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    device1 = torch.device("cpu")

np.random.seed(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(torch.cuda.device_count())
print(device,device1)

def train():

    parser = config_parser()
    args = parser.parse_args()
    # if len(args.skips) == 0:
    #     print('Warning!!!!!!!!!!! No skip connection.')
    writer = SummaryWriter(args.basedir + args.expname)

    print('------------------------------------------')
    ################################################################################
    # Load data
    
    nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny , c, mu_a, mu_s, n, zd = load_data(args.dataset_type, args.datadir)

    if args.n>0:
        n = args.n
        mu_a = args.mu_a
        mu_s = args.mu_s

    R = calculate_reflection_coeff(n)
    ze = 2/3 * 1/mu_s * (1 + R) / (1 - R)
    nlos_data = nlos_data.reshape([Nz,Nx,Ny])

    # nlos_data, camera_grid_positions, deltaT, wall_size ,Nz ,Nx ,Ny = load_data(args.dataset_type, args.datadir)

    # volume_size = np.array([wall_size/2]),np.array([deltaT*Nz/2]),np.array([wall_size/2])
    # volume_position = [0 , deltaT*Nz/2, 0]

    ################################################################################
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    # if args.check_data:
    #     expname = expname + "_test"
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
    img_path = os.path.join(basedir, expname, 'image/')
    loss_path = os.path.join(basedir, expname, 'loss/')
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)

    print(basedir+expname)

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
    ################################################################################
    # Construct our loss function and an Optimizer.
    model = torch.nn.DataParallel(model)
    # model = torch.load("./model_9000.pt")
    # model.eval()
    model = model.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')
    # criterion_sparse = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    ################################################################################
    # ignore some useless bins
    if args.noise:
        nlos_data = nlos_data/nlos_data.max()
        nlos_data = addnoise(nlos_data)
        # nlos_data = add_Gaussian_noise(nlos_data)
        nlos_data = nlos_data/nlos_data.max()*100
    
    if not args.scale:
        nlos_data = nlos_data/100

    data_start = args.neglect_former_nums
    data_end = args.neglect_back_nums
    # nlos_data = nlos_data[data_start:data_end,:]
    nlos_data = torch.Tensor(nlos_data).to(device)
    print(f'All bins < {data_start} and bins > {data_end} are neglected. Ignored data: {nlos_data.shape}')

    # Pre-process
    # pmin = torch.Tensor([-wall_size/2 - data_end * deltaT, - data_end * deltaT, -wall_size/2 - data_end * deltaT]).float().to(device)
    # pmax = torch.Tensor([wall_size/2 + data_end * deltaT, - data_start * deltaT, wall_size/2 + data_end * deltaT]).float().to(device)
    pmin = torch.Tensor([-(wall_size/2) - data_end * deltaT, -1e-7, -(wall_size/2) - data_end * deltaT]).float().to(device)
    pmax = torch.Tensor([wall_size/2 + data_end * deltaT, data_end * deltaT, wall_size/2 + data_end * deltaT]).float().to(device)
    # pmin = torch.Tensor([-wall_size/2 - data_end * deltaT, - data_end * deltaT, -wall_size/2 - data_end * deltaT, 0, -np.pi]).float().to(device)
    # pmax = torch.Tensor([wall_size/2 + data_end * deltaT, - data_start * deltaT, wall_size/2 + data_end * deltaT, np.pi, 0]).float().to(device)

    # Prepare train points and psf
    input_x, input_y, input_z = torch.meshgrid(
        torch.linspace(-wall_size, wall_size, Nx),
        # torch.linspace(-wall_size, wall_size, Nx) + torch.rand(Nx)*(2*wall_size/Nx),
        # torch.linspace(data_start * deltaT, data_end * deltaT, data_end-data_start) **2,
        torch.linspace(data_start * deltaT, data_end * deltaT, data_end-data_start),
        # torch.linspace(data_start * deltaT, data_end * deltaT, data_end-data_start) + torch.rand(data_end-data_start)* deltaT,
        # torch.linspace(data_start * deltaT, data_end * deltaT, data_end-data_start) * -1,
        torch.linspace(-wall_size, wall_size, Ny)
        # torch.linspace(-wall_size, wall_size, Ny)+ torch.rand(Ny)*(2*wall_size/Ny)
    )

    input_x = input_x.reshape([-1, 1])
    input_y = input_y.reshape([-1, 1])
    input_z = input_z.reshape([-1, 1])

    train_input = torch.cat((input_x, input_y, input_z, torch.ones_like(input_x) * np.pi/2, torch.ones_like(input_x) * np.pi/2 * -1), axis = 1).to(device)
    # train_coord = train_input[:,:3].reshape([Nx, data_end-data_start, Ny, 3])
    # train_input = (train_input - pmin) / (pmax - pmin)
    # pdb.set_trace()
    train_input_c, train_input_d = train_input[:,:3], train_input[:,3:]
    train_input_d = Azimuth_to_vector(train_input_d[:,0].reshape([-1,1]), train_input_d[:,1].reshape([-1,1]))
    train_input_c = (train_input_c - pmin)/(pmax - pmin)*2-1 
    # train_input_pe = encoding_sph_tensor(train_input, args.encoding_dim, args.encoding_dim_view,  args.no_rho, device)

    print(f'training input')
    if args.nlos_forward_model=="lct":
        fpsf = psf_for_nlos(wall_size, deltaT, Nx, Ny, Nz, device)
        # resample matrix
        invmtx, invmtxi, _, _ = resamplingOperator(Nz, device)
        psf = fpsf.to(device)
        grid_z = np.repeat(np.repeat(np.expand_dims(np.linspace(0,1,Nz),axis=(1,2)), Nx, axis=1), Ny, axis=2)
        grid_z = torch.from_numpy(grid_z.astype(np.float32)).cuda()

    elif args.nlos_forward_model=="netf":
        kernel = torch.zeros([Nz,Nz,Nx*2-1,Ny*2-1]).to(device)
        input_x, input_y, input_z = torch.meshgrid(
            torch.linspace(0*deltaT, Nz* deltaT, Nz),
            torch.linspace(-wall_size, wall_size, Nx*2-1),
            torch.linspace(-wall_size, wall_size, Ny*2-1),
            indexing='ij'
        )
        dist = input_x**2 + input_y**2 + input_z**2
        # compute the corresponding kernel with different sphere radius
        for j in range(Nz):
            r = (j+1)*deltaT
            r_1 = j*deltaT
            kernel_tmp = torch.zeros([Nz,Nx*2-1,Ny*2-1]).cuda()
            kernel_tmp[dist<=r**2] = 1
            kernel_tmp[dist<=r_1**2] = 0
            # kernel[j-data_start,:,:,:] = kernel_tmp
            kernel[j,:,:,:] = kernel_tmp/(r**4)

        kernel_pad = torch.nn.functional.pad(kernel, [Ny//2-1, Ny//2, Nx//2-1, Nx//2, Nz//2, Nz//2], value=0)
        kernel_rfft = torch.fft.fftn(kernel_pad.reshape(Nz,-1).flip(-1).reshape(Nz, Nz*2, Nx*3-2, Ny*3-2), dim=(1,2,3))

    print(f'nlos kernel done')

    # vol_pad = torch.zeros([2*reso_train, 2*reso_train, 2*reso_train]).to(device)
    # fpsf = fpsf.to(device)
    grid_z = np.repeat(np.repeat(np.expand_dims(np.linspace(0,1,Nz),axis=(1,2)), Nx, axis=1), Ny, axis=2)
    grid_z = torch.from_numpy(grid_z.astype(np.float32))
    
    if args.shift:
        diffuse_psf = set_cdt_completekernel_torch(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, Nz*deltaT*2, zd, device1, n_dipoles = 7)
    else:
        diffuse_psf = set_cdt_completekernel_noshift(Nx, Ny, Nz, c, mu_a, mu_s, ze, wall_size, Nz*deltaT*2, zd, device1, n_dipoles = 7)
    # pdb.set_trace()
    snr = args.snr
    tmp = torch.mul(diffuse_psf, torch.conj(diffuse_psf))
    tmp = tmp + 1/snr
    invpsf = torch.mul(torch.conj(diffuse_psf), 1/tmp)


    # temp = torch.load('./psf.pt')
    # print(torch.sum((temp-diffuse_psf)**2))
    print(f'Pre-precoess done.  diffuse_psf: {diffuse_psf.shape}')

    ################################################################################
    # Prepare log points
    reso = 64
    input_x, input_y, input_z = torch.meshgrid(
        torch.linspace(-wall_size, wall_size, reso),
        torch.linspace(0, Nz*deltaT, reso),
        # torch.linspace(data_start * deltaT, data_end * deltaT, reso),
        torch.linspace(-wall_size, wall_size, reso)
    )

    input_x = input_x.reshape([-1, 1])
    input_y = input_y.reshape([-1, 1])
    input_z = input_z.reshape([-1, 1])
    test_input = torch.cat((input_x, input_y, input_z, torch.ones_like(input_x) * np.pi/2, torch.ones_like(input_x) * np.pi/2 * -1), axis = 1)
    # test_input = (test_input - pmin) / (pmax - pmin)
    test_input_c, test_input_d = test_input[:,:3], test_input[:,3:]
    test_input_d = Azimuth_to_vector(test_input_d[:,0].reshape([-1,1]), test_input_d[:,1].reshape([-1,1]))
    test_input_c = (test_input_c - pmin)/(pmax - pmin)*2-1 

    # test_input_pe = encoding_sph_tensor(test_input, args.encoding_dim, args.encoding_dim_view,  args.no_rho, device)

    N_iters = args.N_iters
    lr_decay_rate = args.lr_decay_rate
    i_loss = args.i_loss
    i_image = args.i_image
    i_model = args.i_model
    i_print = args.i_print

    snr = args.snr
    threshold = nlos_data.max()/20
    threshold = 10
    print('Loss threshold:',threshold)
    print('------------------------------------------')
    print('Training Begin')

    global_step = 0
    loss_global = []
    time0 = time.time()
    for i in trange(0, N_iters):
        ################################################################################transient
        # pdb.set_trace()
        sigma, color = model(train_input_c, train_input_d)
        network_res = torch.mul(sigma, color)
        network_res = network_res.reshape([Nx, data_end-data_start, Ny])

        if args.nlos_forward_model=="lct":
            vol_pad = F.pad(network_res, [0, 0, data_start, Nz-data_end, 0, 0]).permute([1, 0, 2])
            # print('vol:'vol_pad.shape)
            # vol_pad = network_res.permute([1, 0, 2])
            vol_resample = torch.mm(invmtxi,vol_pad.reshape(Nz,-1)).reshape([Nz, Nx, Ny])
            # vol_resample = torch.load('./test.npy')
            # vol_resample = network_res.permute([1, 0, 2])

            # convolution
            vol_fft = torch.fft.fftn(vol_resample.to(device1),s=(Nz*2, Nx*2, Ny*2))
            pre_trans = torch.fft.ifftn(torch.mul(vol_fft.to(device1), fpsf.to(device1)),s=(Nz*2, Nx*2, Ny*2)).real

            # unpad and resample transient
            pre_trans = pre_trans[:Nz,:Nx,:Ny].to(device)
            pre_trans = torch.mm(invmtx,pre_trans.reshape(Nz,-1)).reshape([Nz, Nx, Ny])
            pre_trans = pre_trans.to(device1)/(grid_z.to(device1)**4+1e-8)
            # pre_trans = pre_trans/(grid_z**4+1e-8)
            pre_trans = pre_trans[data_start:data_end,...]
            
        elif args.nlos_forward_model=="netf":
            vol_pad = F.pad(network_res, [0, 0, data_start, Nz-data_end, 0, 0]).permute([1, 0, 2])
            vol_pad = torch.nn.functional.pad(vol_pad, [Nx-1, Nx-1, Nx-1, Nx-1, Nz//2, Nz//2], value=0)
            vol_rfft = torch.fft.fftn(vol_pad).cuda()
            pre_trans = torch.fft.ifftshift(torch.fft.ifftn(torch.mul(vol_rfft, kernel_rfft), dim=(1,2,3)), dim=(1,2,3)).real[:,Nz,Nx-1:Nx*2-1,Nx-1:Nx*2-1]
            pre_trans = pre_trans[data_start:data_end,...]

        nlos_pad = F.pad(pre_trans, [0, 0, 0, 0, data_start, Nz-data_end] )
        # pdb.set_trace()

        # if args.check_data:
        #     nlos_pad = torch.Tensor(tmp_data)[None, None, :, :, :]
        if args.shift:
            # nlos_pad = F.pad(nlos_pad, [0, Ny-1, 0, Nx-1, 0, Nz] )[None, None, :, :, :]
            cdt_conv = torch.mul(torch.fft.fftn(nlos_pad,s=(Nz*2, Nx*2-1, Ny*2-1)), diffuse_psf)
        else:
            # nlos_pad = F.pad(nlos_pad, [0, Ny, 0, Nx, 0, Nz] )[None, None, :, :, :]
            cdt_conv = torch.mul(torch.fft.fftn(nlos_pad,s=(Nz*2, Nx*2, Ny*2)), diffuse_psf)

        predict_cdt = torch.fft.ifftn(cdt_conv).real.squeeze()
        # print('predict_cdt:',predict_cdt.min(),predict_cdt.max())


        cdt_pad = torch.clone(predict_cdt)
        # cdt_pad = torch.zeros_like(predict_cdt)
        cdt_pad[:Nz,:Nx,:Ny] = nlos_data
        # pdb.set_trace()

        # snr = np.exp(i//1000+20)
        # if snr > 1e10:
        #     snr = 1e10
        # tmp = torch.mul(diffuse_psf, torch.conj(diffuse_psf))
        # tmp = tmp + 1/snr
        # invpsf = torch.mul(torch.conj(diffuse_psf), 1/tmp)
        

        if args.shift:
            cdt_pad_fft = torch.fft.fftn(cdt_pad.to(device), s=(Nz*2, Nx*2-1, Ny*2-1))                  
        else:
            # print("Magic here.")
            cdt_pad_fft = torch.fft.fftn(cdt_pad.to(device), s=(Nz*2, Nx*2, Ny*2))
            
            # cdt_pad_fft = torch.fft.fftn(cdt_pad, s=(Nz*2, Nx*2, Ny*2))
            # temp = torch.load('./x_rfft.pt').cuda()
            # diff = (temp-cdt_pad_fft.real)**2
            # print(diff.min(), diff.max(), diff.mean())
            # print(torch.sum(diff))
            # print("Magic done.")

        # target_nlos = torch.fft.fftn(torch.mul(cdt_pad_fft, invpsf)).real
        target_fft = torch.mul(cdt_pad_fft.to(device), invpsf.to(device))
        target_nlos = torch.fft.ifftn(target_fft).real.squeeze()[:Nz,:Nx,:Ny]
        # temp = torch.load('x_deconv.pt').cuda()
        # print(torch.sum((temp-target_nlos)**2))
        # temp = torch.load('./psf.pt').cuda()
        # print(torch.sum((temp-invpsf.real)**2))

        # predict_histogram = predict_cdt[index_rand,...]
        # nlos_histogram = nlos_data[index_rand,...]
        # predict_histogram = predict_cdt[data_start:data_end,:,:]
        predict_histogram = predict_cdt[data_start:data_end,:Nx,:Ny]
        nlos_histogram = nlos_data[data_start:data_end,...]
        # data_pred = torch.nn.functional.conv3d(vol_pad.reshape([1,1,reso_train*2,reso_train*2,reso_train*2]), psf.reshape([data_end-data_start,1,reso_train,reso_train,reso_train]), stride=conv_stride)
        # data_pred = data_pred[0,:,:-1,0,:-1]
        # data_pred = data_pred.reshape([data_end-data_start,-1])
        # nlos_pad = torch.zeros([Nz, Nx, Ny]).cuda()
        # nlos_pad[data_start:data_end,:,:] = data_pred
        # nlos_pad = data_pred
        # cdt_conv = torch.fft.fftn(torch.mul(torch.fft.fftn(nlos_pad , s=(Nz*2, Nx*2-1, Ny*2-1)), diffusion_fpsf)).real
        # predict_cdt = cdt_conv.squeeze()[:Nz,:Nx,:Ny]
        # predict_cdt = cdt_conv.squeeze()[data_start:data_end,:Nx,:Ny]
        # predict_cdt = predict_cdt.reshape([Nz,-1])
        # predict_cdt = data_pred
        # predict_cdt = predict_cdt.reshape([data_end-data_start,-1])
        # predict_cdt = data_pred

        # update
        loss = criterion(predict_histogram.to(device1), nlos_histogram.to(device1))
        # loss = criterion(data_pred, nlos_data)

        # pre_trans_cat_0 = torch.cat((pre_trans[:,1:,:], pre_trans[:,-1,:].unsqueeze(1)), 1)
        # pre_trans_cat_1 = torch.cat((pre_trans[:,:,1:], pre_trans[:,:,-1].unsqueeze(2)), 2)
        # tv_loss = torch.sum((pre_trans - pre_trans_cat_0)**2+(pre_trans - pre_trans_cat_1)**2)
        # cauchy_loss = torch.sum(torch.log(1+2*(network_res)**2))
        # lamb = 1e-5
        # sparse_loss = torch.sum(abs(1-torch.exp(-1*lamb*pre_trans)))
        # # loss = loss + sparse_loss
        # loss = loss + lamb*cauchy_loss
        # trim = Nx*Ny*Nz//20
        trim = args.trim
        # loss1 = criterion(nlos_pad.squeeze()[:-trim], target_nlos[:-trim])
        target_nlos = target_nlos.to(device1)
        nlos_pad = nlos_pad.to(device1)
        if args.nlos_neglect_former_bins:
            target_nlos = target_nlos[data_start:data_end,...]
            nlos_pad = nlos_pad[data_start:data_end,...]

        loss1 = criterion(nlos_pad.squeeze()[target_nlos>=0][:-trim], target_nlos[target_nlos>=0][:-trim])

        # if loss<threshold:
        #    loss = loss1

        # loss = loss1
        if args.cdt_loss == "deconv":
            loss = loss1
        elif args.cdt_loss == "both":
            loss = loss1+loss
            
        # if args.check_data:
        #     loss = loss
        # if not args.check_data:
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

            # tmp = F.pad(nlos_data,[0,Ny,0,Nx,0,Nz])
            plt.plot(nlos_histogram.cpu().detach().numpy().flatten(), alpha = 0.5, label = 'data')
            plt.plot(predict_histogram.cpu().detach().numpy().flatten(), alpha = 0.5, label='predicted')
            # plt.plot(tmp_data.squeeze()[data_start:data_end,...].flatten(), alpha = 0.5, label = 'nlos')

            # plt.plot(tmp.cpu().detach().numpy().flatten(), alpha = 0.5, label = 'data')
            # plt.plot(target_fft.real.cpu().detach().numpy().flatten(), alpha = 0.5, label = 'pad')
            # plt.plot(torch.fft.fftn(nlos_pad,s=(Nz*2, Nx*2, Ny*2)).real.cpu().detach().numpy().flatten(), alpha = 0.5, label='predicted_fft')

            # plt.plot(nlos_pad.squeeze()[:-trim].cpu().detach().numpy().flatten(), alpha = 0.5, label = 'nlos')
            # plt.plot(tmp_data[:-trim].flatten(), alpha = 0.5, label = 'nlos_data')
            # plt.plot(target_nlos[:-trim].cpu().detach().numpy().flatten(), alpha = 0.5, label = 'target_nlos')  

            if not args.cdt_loss=="cdt":
                plt.plot(nlos_pad.squeeze()[target_nlos>=0][:-trim].cpu().detach().numpy().flatten(), alpha = 0.5, label = 'pred_nlos')
                # if args.check_data:
                #     plt.plot(tmp_data[target_nlos.cpu()>=0][:-trim].flatten(), alpha = 0.5, label = 'nlos_data')
                plt.plot(target_nlos[target_nlos>=0][:-trim].cpu().detach().numpy().flatten(), alpha = 0.5, label = 'target_nlos')       
            #      
            # plt.plot(tmp_data.flatten(), alpha = 0.5, label = 'nlos_data')
            # plt.plot(nlos_pad.squeeze().cpu().detach().numpy().flatten(), alpha = 0.5, label = 'nlos')
            # plt.plot(target_nlos.cpu().detach().numpy().flatten(), alpha = 0.5, label = 'target_nlos')
            plt.legend()
            plt.savefig(loss_path + str(i+1) + 'histogram')
            plt.close()

        # log recon result
        if (i+1) % i_image == 0:
            with torch.no_grad():
                sigma, color = model(test_input_c, test_input_d)
                output = torch.mul(sigma, color)
                # output = model(train_input_pe)
                # temp = (output[:,0] * output[:,1]).reshape([Nx, Nz, Ny])
                # temp = output.reshape([Nx, data_end-data_start, Ny])
                temp = output.reshape([reso, reso, reso])
                # temp[:,-10:,:] = 0
                temp[:,-10:,:] = 0
                temp[:,:data_start*reso//Nz,:] = 0
                temp[:,data_end*reso//Nz:,:] = 0
                # temp = F.pad(temp,[0, 0, data_start*reso//(data_end-data_start), reso*(Nz-data_end)//(data_end-data_start),0, 0])
                # temp = (output[:,0] * output[:,1]).reshape([Nx, data_end-data_start, Ny])
                # output = model(test_input_pe)
                # temp = (output[:,0] * output[:,1]).reshape([reso, reso, reso])
                # temp = (output[:,0] * output[:,1]).reshape([Nx, data_end-data_start, Ny])
                # temp = torch.fft.fftn(temp).real
                # temp = F.relu(temp)
                temp = temp/temp.max()
                # temp[:,-10:,:] = 0
                # temp[:,:data_start*reso//Nz,:] = 0
                # temp[:,data_end*reso//Nz:,:] = 0
                # temp = pre_trans.cpu().data

                temp_img = temp.max(axis = 1).values
                plt.imsave(img_path + 'result_' + str(i+1) + '_XOY.png', temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.close()

                temp_img = temp.max(axis = 0).values
                plt.imsave(img_path + 'result_' + str(i+1) + '_YOZ.png', temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.close()

                temp_img = temp.max(axis = 2).values
                plt.imsave(img_path + 'result_' + str(i+1) + '_XOZ.png', temp_img.cpu().data.numpy().squeeze(), cmap='gray')
                plt.close()
        
        # log model
        if (i+1) % i_model == 0:
            model_name = model_path + 'model_' + str(i+1) + '.pt'
            torch.save(model, model_name)

        # log print
        if (i+1) % i_print == 0: 
            dt = time.time()-time0
            tqdm.write(f"[TRAIN] Iter: {i+1} / {N_iters}, Loss: {(loss.item())}, Loss1: {loss1.item()}, lrate: {new_lrate}, TIME: {dt}")
            # tqdm.write(f"[TRAIN] Iter: {i+1} / {N_iters}, Loss: {loss.item()}, lrate: {new_lrate}, TIME: {dt}")
            time0 = time.time()

if __name__=='__main__':
    # python run_netf.py --config configs/nlos.txt
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()