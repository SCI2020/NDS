import configargparse

def config_parser():
    
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
    parser.add_argument("--scale_data", action='store_true', 
                        help='when True, multiply the normalized data by 100')
    parser.add_argument("--shift", action='store_true', 
                        help='when True, the data has been shifted')
    # MLP options
    parser.add_argument("--tcnn", action='store_true',
                        help='use tcnn model or not')    
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
    parser.add_argument("--bbox", action='store_true', default= False,
                        help='use bounding box or not')  
    parser.add_argument("--occlusion", action='store_true', default= False,
                        help='use occlusion in forward model or not')  
    parser.add_argument("--loss", type=str, default='mse', 
                        help='encoding type for position')
    # parser.add_argument("--snr", type=float, default=1e10, 
    #                     help='learning rate decay rate')
    
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
    parser.add_argument("--i_obj", type=int, default=100, 
                        help='num of iters to log obj') 
    parser.add_argument("--obj_threshold", type=float, default=0.01, 
                        help='threshold for obj extraction')    
    
    # cdt options 
    parser.add_argument("--cdt_loss", type=str, default='cdt', 
                        help='when True, the data has been shifted')
    parser.add_argument("--snr", type=float, default=1e10, 
                        help='learning rate decay rate')
    parser.add_argument("--trim", type=int, default=5000, 
                        help='num of iters to log print') 
    parser.add_argument("--nlos_neglect_former_bins", action='store_true', default = False, 
                        help='when True, those former histogram bins will be neglected and not used in optimization. The threshold is computed automatically to ensure that neglected bins are zero')
    parser.add_argument("--noise", action='store_true', default = False , 
                        help='add noise to data or not')
    parser.add_argument("--nlos_forward_model", type=str, default='lct', 
                        help='input data directory')
    parser.add_argument("--scale", action='store_true', default=True,
                        help='when True, the data has been shifted')
    return parser
