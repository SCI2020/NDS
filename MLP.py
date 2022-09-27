# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder

class Network(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, skips=[4], no_rho=False):

        super(Network, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.no_rho = no_rho

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])

        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)])

        if self.no_rho:
            self.output_linear = torch.nn.Linear(H, 1)
        else:
            self.feature_linear = torch.nn.Linear(H, H)
            self.alpha_linear = torch.nn.Linear(H, 1)
            self.rho_linear = torch.nn.Linear(H//2, 1)

    def forward(self, x):

        if self.no_rho:
            input_pts = x
            h = x
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.no_rho:
            outputs = self.output_linear(h)
            outputs = torch.abs(outputs)
        else:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.nn.functional.relu(h)

            rho = self.rho_linear(h)
            alpha = torch.abs(alpha)
            rho = torch.abs(rho)
            outputs = torch.cat([rho, alpha], -1)

        return outputs

class NGPNetwork(torch.nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 reso=64
                 ):
        # super().__init__(bound, **kwargs)
        super(NGPNetwork, self).__init__()
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=64 * bound)
        
        self.bound = bound

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 1 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        # sigma = torch.abs(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        # color = torch.sigmoid(h)
        # color = torch.abs(h)
        color = F.relu(h)

        return sigma.reshape([-1,1]), color


# if __name__ == "__main__": # used for test MLP
#     seed = 0
#     torch.manual_seed(seed)            # 为CPU设置随机种子
#     torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
