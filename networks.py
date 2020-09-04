# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================================paddorch=========
# import porch
# import porch.nn as nn


import paddorch as porch
import paddorch.nn as  nn
from paddorch.nn.parameter import Parameter

class AdaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter( (1, num_features, 1, 1),0.9) #porch.zeros((1, num_features, 1, 1))+0.9 #Parameter(porch.Tensor(1, num_features, 1, 1))
        # self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        in_mean, in_var = porch.mean(x, dim=[2, 3], keepdim=True), porch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / porch.sqrt(in_var + self.eps)
        ln_mean, ln_var = porch.mean(x, dim=[1, 2, 3], keepdim=True), porch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / porch.sqrt(ln_var + self.eps)
        out = porch.Tensor(self.rho).expand(x.shape[0], -1, -1, -1) * out_in + (1 - porch.Tensor(self.rho).expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * porch.Tensor(gamma).unsqueeze(2).unsqueeze(3) + porch.Tensor(beta).unsqueeze(2).unsqueeze(3)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2, 0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 1, 0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, 1, 1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, inputs):
        x = self.model(inputs)

        gap = porch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * porch.Tensor(gap_weight).unsqueeze(0).unsqueeze(3)

        gmp = porch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * porch.Tensor(gmp_weight).unsqueeze(0).unsqueeze(3)

        cam_logit = porch.cat([gap_logit, gmp_logit], 1)
        x = porch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = porch.sum(x, dim=1, keepdim=True)
        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit,heatmap


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.n_res=n_blocks
        self.light= light
        down_layer = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf,affine=True),
            nn.ReLU(inplace=True),

            # Down-Sampling
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf*2, 3, 2, 0, bias=False),
            nn.InstanceNorm2d(ngf*2,affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 0, bias=False),
            nn.InstanceNorm2d(ngf*4,affine=True),
            nn.ReLU(inplace=True),

            # Down-Sampling Bottleneck
            ResNetBlock(ngf*4),
            ResNetBlock(ngf*4),
            ResNetBlock(ngf*4),
            ResNetBlock(ngf*4),
        ]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf*4, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf*4, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf*8, ngf*4, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # # Gamma, Beta block
        # fc = [
        #     nn.Linear(image_size * image_size * 16, 256, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 256, bias=False),
        #     nn.ReLU(inplace=True)
        # ]
        # Gamma, Beta block
        if self.light:
            fc = [nn.Linear(ngf*4, ngf*4, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf*4, ngf*4, bias=False),
                  nn.ReLU(True)]
        else:
            fc = [nn.Linear(img_size * img_size * ngf//4, ngf*4, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf*4, ngf*4, bias=False),
                  nn.ReLU(True)]


        self.gamma = nn.Linear(ngf*4, ngf*4, bias=False)
        self.beta = nn.Linear(ngf*4, ngf*4, bias=False)

        # Up-Sampling Bottleneck
        for i in range(self.n_res):
            setattr(self, "ResNetAdaILNBlock_" + str(i + 1), ResNetAdaILNBlock(ngf*4))

        up_layer = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf*4, ngf*2, 3, 1, 0, bias=False),
            ILN(ngf*2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf*2, ngf, 3, 1, 0, bias=False),
            ILN(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7, 1, 0, bias=False),
            nn.Tanh()
        ]

        self.down_layer = nn.Sequential(*down_layer)
        self.fc = nn.Sequential(*fc)
        self.up_layer = nn.Sequential(*up_layer)

    def forward(self, inputs):
        x = self.down_layer(inputs)

        gap = porch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = porch.Tensor(list(self.gap_fc.parameters())[0]).permute(1,0)
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = porch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight =  porch.Tensor(list(self.gmp_fc.parameters())[0]).permute(1,0)
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)


        cam_logit = porch.cat([gap_logit, gmp_logit], 1)
        x = porch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        x =porch.Tensor(x)
        heatmap = porch.sum(x, dim=1, keepdim=True)
        if self.light:
            x_ = porch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.fc(x_.view(x_.shape[0], -1))
        else:
            x_ = self.fc(x.view(x.shape[0], -1))



        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_res):
            x = getattr(self, "ResNetAdaILNBlock_" + str(i + 1))(x, gamma, beta)
        out = self.up_layer(x)

        return out, cam_logit ,heatmap


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter( (1, num_features, 1, 1),0.0) #  Parameter(porch.Tensor(1, num_features, 1, 1))
        self.gamma =Parameter( (1, num_features, 1, 1),1.0) # Parameter(porch.Tensor(1, num_features, 1, 1))
        self.beta =Parameter( (1, num_features, 1, 1),0.0) # Parameter(porch.Tensor(1, num_features, 1, 1))


    def forward(self, x):
        in_mean, in_var = porch.mean(x, dim=(2, 3), keepdim=True), porch.var(x, dim=(2, 3), keepdim=True)
        out_in = (x - in_mean) / porch.sqrt(in_var + self.eps)
        ln_mean, ln_var = porch.mean(x, dim=(1, 2, 3), keepdim=True), porch.var(x, dim=(1, 2, 3), keepdim=True)
        out_ln = (x - ln_mean) / porch.sqrt(ln_var + self.eps)
        out = porch.Tensor(self.rho).expand(x.shape[0], -1, -1, -1) * out_in + (1 - porch.Tensor(self.rho).expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * porch.Tensor(self.gamma).expand(x.shape[0], -1, -1, -1) + porch.Tensor(self.beta).expand(x.shape[0], -1, -1, -1)

        return out


class ResNetAdaILNBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 0, bias=False)
        self.norm1 = AdaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 0, bias=False)
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
                       nn.InstanceNorm2d(dim,affine=True),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
                       nn.InstanceNorm2d(dim,affine=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class RhoClipper(object):

    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max
        assert clip_min < clip_max

    def __call__(self, module):
        if hasattr(module, "rho"):
            w = porch.Tensor(module.rho)
            w = w.clamp_(self.clip_min, self.clip_max)
            module.rho = w



