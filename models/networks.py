import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG,
             norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'downandupsample':
        net = DownAndUpsample(input_nc, output_nc, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not reconized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = DownsampleDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    return init_net(net, init_type, init_gain, gpu_ids)



##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    """
    Defines the GAN loss which uses either LSGAN or the regular GAN.
    When LSGAN is used, it is basically same as MSELoss,
    but it abstracts away the need to create the target lagel tensor
    that has the same size as the input
    """
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DownAndUpsample(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.BatchNorm2d):
        super(DownAndUpsample, self).__init__()
        #norm_layer = 
        # padding == (kernel_size - stride) // 2
        self.model = nn.Sequential(
            ## Downsampling
            # 1st conv layer
            nn.Conv2d(input_nc, 2 * ngf, kernel_size=[3, 9], stride=[1, 1],
                      padding=[1, 4], bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.GLU(dim=1),
            # 2nd conv layer
            nn.Conv2d(ngf, 2 * ngf * 2, kernel_size=[4, 8], stride=[2, 2],
                      padding=[1, 3], bias=False),
            nn.BatchNorm2d(2 * ngf * 2),
            nn.GLU(dim=1),
            # 3rd conv layer
            nn.Conv2d(ngf * 2, 2 * ngf * 4, kernel_size=[4, 8], stride=[2, 2],
                      padding=[1, 3], bias=False),
            nn.BatchNorm2d(2 * ngf * 4),
            nn.GLU(dim=1),
            # 4th conv layer
            nn.Conv2d(ngf * 4, 2 * ngf * 2, kernel_size=[3, 5], stride=[1, 1],
                      padding=[1, 2], bias=False),
            nn.BatchNorm2d(2 * ngf * 2),
            nn.GLU(dim=1),
            # 5th conv layer
            nn.Conv2d(ngf * 2, 2 * 5, kernel_size=[10, 5], stride=[10, 1],
                      padding=[0, 2], bias=False),
            nn.BatchNorm2d(2 * 5),
            nn.GLU(dim=1),

            ## Upsampling
            # 1st deconv layer
            nn.ConvTranspose2d(5, 2 * ngf * 2, kernel_size=[10, 5], stride=[10, 1],
                                padding=[0, 2], bias=False),
            nn.BatchNorm2d(2 * ngf * 2),
            nn.GLU(dim=1),
            # 2nd deconv layer
            nn.ConvTranspose2d(ngf * 2, 2 * ngf * 4, kernel_size=[3, 5], stride=[1, 1],
                                padding=[1, 2], bias=False),
            nn.BatchNorm2d(2 * ngf * 4),
            nn.GLU(dim=1),
            # 3rd deconv layer
            nn.ConvTranspose2d(ngf * 4, 2 * ngf * 2, kernel_size=[4, 8], stride=[2, 2],
                                padding=[1, 3], bias=False),
            nn.BatchNorm2d(2 * ngf * 2),
            nn.GLU(dim=1),
            # 4th deconv layer
            nn.ConvTranspose2d(ngf * 2, 2 * ngf, kernel_size=[4, 8], stride=[2, 2],
                                padding=[1, 3], bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.GLU(dim=1),
            # 5th deconv layer
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=[3, 9], stride=[1, 1],
                                padding=[1, 4], bias=False))

    def forward(self, input):
        return self.model(input)


class DownsampleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=32, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(DownsampleDiscriminator, self).__init__()
        # padding == (kernel_size - stride) // 2
        self.model = nn.Sequential(
            ## Downsampling
            # 1st conv layer
            nn.Conv2d(input_nc, 2 * ndf, kernel_size=[3, 9], stride=[1, 1],
                      padding=[1, 4], bias=False),
            nn.BatchNorm2d(2 * ndf),
            nn.GLU(dim=1),
            # 2nd conv layer
            nn.Conv2d(ndf, 2 * ndf, kernel_size=[3, 8], stride=[1, 2],
                      padding=[1, 3], bias=False),
            nn.BatchNorm2d(2 * ndf),
            nn.GLU(dim=1),
            # 3rd conv layer
            nn.Conv2d(ndf, 2 * ndf, kernel_size=[3, 8], stride=[1, 2],
                      padding=[1, 3], bias=False),
            nn.BatchNorm2d(2 * ndf),
            nn.GLU(dim=1),
            # 4th conv layer
            nn.Conv2d(ndf, 2 * ndf, kernel_size=[3, 6], stride=[1, 2],
                      padding=[1, 2], bias=False),
            nn.BatchNorm2d(2 * ndf),
            nn.GLU(dim=1),
            # 5th conv layer
            nn.Conv2d(ndf, 1, kernel_size=[80, 5], stride=[80, 1],
                      padding=[0, 2], bias=False),
            nn.Sigmoid())
    
    def forward(self, input):
        output = self.model(input)
        output = torch.mean(output, dim=-1, keepdim=True)
        output = output.reshape(-1,1)
        return output