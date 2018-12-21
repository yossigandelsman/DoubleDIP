import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_io import np_to_torch


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 100):
    """
    Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth % 2 == 0
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]] * (input_depth // 2))
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def get_video_noise(input_depth, method, temporal_size, spatial_size, noise_type='u', var=1. / 100, type="dependant"):
    """
    Returns a pytorch.Tensor of size (frame_number x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        temporal_size: number of frames
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        all_noise = []
        for i in range(temporal_size):
            shape = [input_depth, spatial_size[0], spatial_size[1]]
            if len(all_noise) > 0:
                if type == "dependant":
                    frame = np.random.uniform(0, 1, size=shape)
                    frame *= var
                    all_noise.append(all_noise[-1] + frame)
                elif type == "half_dependant":
                    frame = np.random.uniform(0, 1, size=shape)
                    frame *= var
                    new_noise = (all_noise[-1] + frame)
                    new_noise[:input_depth // 2,:,:] = (var * 10) * np.random.uniform(0, 1, size=shape)[:input_depth // 2,:,:]
                    all_noise.append(new_noise)
            else:
                frame = np.random.uniform(-0.5, 0.5, size=shape)
                frame *= (var * 10)
                all_noise.append(frame)
        return np_to_torch(np.array(all_noise))[0]
    elif method == 'meshgrid':
        assert False
        assert input_depth % 2 == 0
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]] * (input_depth // 2))
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


class NoiseNet(nn.Module):
    def __init__(self, channels=3, kernel_size=5):
        super(NoiseNet, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        to_pad = int((self.kernel_size - 1) / 2)
        self.padder = nn.ReflectionPad2d(to_pad).type(torch.cuda.FloatTensor)
        to_pad = 0
        self.convolver = nn.Conv2d(channels, channels, self.kernel_size, 1, padding=to_pad, bias=True).type(torch.cuda.FloatTensor)

    def forward(self, x):
        assert x.shape[1] == self.channels, (x.shape, self.channels)
        first = F.relu(self.convolver(self.padder(x)))
        second = F.relu(self.convolver(self.padder(first)))
        third = F.relu(self.convolver(self.padder(second)))
        assert x.shape == third.shape, (x.shape, third.shape)
        return third


def fill_noise(x, noise_type):
    """
    Fills tensor `x` with noise of type `noise_type`.
    """
    if noise_type == 'u':
        x.uniform_(-0.5, 0.5)
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False