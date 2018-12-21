from net import skip
from net.losses import ExclusionLoss, plot_image_grid, StdLoss
from net.noise import get_noise
from utils.image_io import *
from skimage.measure import compare_psnr
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

TwoImagesSeparationResult = namedtuple("TwoImagesSeparationResult",
                                       ["reflection", "transmission", "psnr", "alpha1", "alpha2"])

class TwoImagesSeparation(object):
    def __init__(self, image1_name, image2_name, image1, image2, plot_during_training=True, show_every=500, num_iter=4000,
                 original_reflection=None, original_transmission=None):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 2
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.image1_torch = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(self.image2).type(torch.cuda.FloatTensor)

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        self.reflection_net_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.alpha_net1_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.alpha_net2_input = get_noise(self.input_depth, input_type,
                                          (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.transmission_net_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]
        self.parameters += [p for p in self.alpha1.parameters()]
        self.parameters += [p for p in self.alpha2.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)
        alpha_net1 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha1 = alpha_net1.type(data_type)

        alpha_net2 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha2 = alpha_net2.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self, step):
        reg_noise_std = 0
        reflection_net_input = self.reflection_net_input + (self.reflection_net_input.clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_input + \
                                 (self.transmission_net_input.clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)
        self.transmission_out = self.transmission_net(transmission_net_input)
        alpha_net_input = self.alpha_net1_input + (self.alpha_net1_input.clone().normal_() * reg_noise_std)
        self.current_alpha1 = self.alpha1(alpha_net_input)[:, :,
                             self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                             self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05

        alpha_net_input = self.alpha_net2_input + (self.alpha_net2_input.clone().normal_() * reg_noise_std)
        self.current_alpha2 = self.alpha2(alpha_net_input)[:, :,
                              self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                              self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1]* 0.9 + 0.05
        self.total_loss = self.mse_loss(self.current_alpha1 * self.reflection_out +
                                        (1 - self.current_alpha1) * self.transmission_out,
                                        self.image1_torch)
        self.total_loss += self.mse_loss(self.current_alpha2 * self.reflection_out +
                                         (1 - self.current_alpha2) * self.transmission_out,
                                         self.image2_torch)
        self.exclusion = self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss += 0.1 * self.exclusion
        # self.total_loss += self.blur_loss(self.current_alpha2) + self.blur_loss(self.current_alpha1)
        if step < 1000:
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha1,
                                                  torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha2,
                                                  torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))

        self.total_loss.backward()

    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
        transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
        # print(reflection_out_np.shape)
        alpha1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        alpha2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)
        v = alpha1 * reflection_out_np + (1 - alpha1) * transmission_out_np
        # print(v.shape, self.image2.shape)
        psnr1 = compare_psnr(self.image1, v)
        psnr2 = compare_psnr(self.image2, alpha2 * reflection_out_np + (1 - alpha2) * transmission_out_np)
        self.psnrs.append(psnr1+psnr2)
        self.current_result = TwoImagesSeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                        psnr=psnr1, alpha1=alpha1, alpha2=alpha2)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Exclusion {:5f}  PSRN_gt: {:f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                                    self.exclusion.item(),
                                                                                    self.current_result.psnr),
              '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(step),
                            [self.current_result.reflection, self.current_result.transmission])
            # plot_image_grid("learned_mask_{}".format(step),
            #                 [self.current_result.alpha1, self.current_result.alpha2])
            save_image("sum1_{}".format(step), self.current_result.alpha1 * self.current_result.reflection +
                       (1-self.current_result.alpha1)* self.current_result.transmission)
            save_image("sum2_{}".format(step), self.current_result.alpha2 * self.current_result.reflection +
                       (1 - self.current_result.alpha2) * self.current_result.transmission)

    def finalize(self):
        save_graph(self.image1_name + "_psnr", self.psnrs)
        save_image(self.image1_name + "_reflection", self.best_result.reflection)
        save_image(self.image1_name + "_transmission", self.best_result.transmission)
        save_image(self.image1_name + "_original", self.image1)
        save_image(self.image2_name + "_original", self.image2)


class Separation(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter=8000,
                 original_reflection=None, original_transmission=None):
        self.image = image
        self.plot_during_training = plot_during_training
        # self.ratio = ratio
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        # self.ratio_net = None
        self.parameters = None
        self.learning_rate = 0.0005
        self.input_depth = 3
        self.reflection_net_inputs = None
        self.transmission_net_inputs = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images = create_augmentations(self.image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                                  input_type,
                                                  (self.images_torch[0].shape[2],
                                                   self.images_torch[0].shape[3])).type(data_type).detach())
        self.reflection_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in create_augmentations(origin_noise)]
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.transmission_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                      create_augmentations(origin_noise)]

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.exclusion_loss =  ExclusionLoss().type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        # return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(step)
        if step == self.num_iter - 1:
            aug = 0
        reflection_net_input = self.reflection_net_inputs[aug] + (self.reflection_net_inputs[aug].clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_inputs[aug] + (self.transmission_net_inputs[aug].clone().normal_() * reg_noise_std)


        self.reflection_out = self.reflection_net(reflection_net_input)

        self.transmission_out = self.transmission_net(transmission_net_input)

        self.total_loss = self.l1_loss(self.reflection_out + self.transmission_out, self.images_torch[aug])
        self.total_loss += 0.01 * self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.num_iter - 1 or step % 8 == 0:
            reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
            transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
            psnr = compare_psnr(self.images[0],  reflection_out_np + transmission_out_np)
            self.psnrs.append(psnr)
            self.current_result = SeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                   psnr=psnr)
            if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}  PSRN_gt: {:f}'.format(step,
                                                                               self.total_loss.item(),
                                                                               self.current_result.psnr),
              '\r', end='')
        if step % self.show_every == self.show_every - 1:
            plot_image_grid("left_right_{}".format(step), [self.current_result.reflection, self.current_result.transmission])
            
    def _plot_distance_map(self):
        calculated_left = self.best_result.reflection
        calculated_right = self.best_result.transmission
        # this is to left for reason
        # print(distance_to_left.shape)
        pass

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_image(self.image_name + "_reflection", self.best_result.reflection)
        save_image(self.image_name + "_transmission", self.best_result.transmission)
        save_image(self.image_name + "_reflection2", 2 * self.best_result.reflection)
        save_image(self.image_name + "_transmission2", 2 * self.best_result.transmission)
        save_image(self.image_name + "_original", self.images[0])



SeparationResult = namedtuple("SeparationResult", ['reflection', 'transmission', 'psnr'])

if __name__ == "__main__":
    # Separation from two images
    input1 = prepare_image('images/input1.jpg')
    input2 = prepare_image('images/input2.jpg')
    t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=7000)
    t.optimize()
    t.finalize()
    # Separation of textures
    t1 = prepare_image('images/texture12.jpg')
    t2 = prepare_image('images/texture16.jpg')
    s = Separation('textures', (t1+t2)/2)
    s.optimize()
    s.finalize()
