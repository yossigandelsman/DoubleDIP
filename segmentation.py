from collections import namedtuple

from net import *
from net.losses import StdLoss, YIQGNGCLoss, GradientLoss, ExtendedL1Loss, GrayLoss
from net.noise import get_noise, NoiseNet
from utils.image_io import *
from net.downsampler import *
from skimage.measure import compare_psnr
from cv2.ximgproc import guidedFilter

SegmentationResult = namedtuple("SegmentationResult", ['mask', 'learned_mask', 'left', 'right', 'psnr'])


class Segmentation(object):
    def __init__(self, image_name, image, plot_during_training=True,
                 first_step_iter_num=2000,
                 second_step_iter_num=4000,
                 bg_hint=None, fg_hint=None,
                 show_every=500,
                 downsampling_factor=0.1, downsampling_number=0):
        self.image = image
        if bg_hint is None or fg_hint is None: 
            raise Exception("Hints must be provided")
        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.downsampling_factor = downsampling_factor
        self.downsampling_number = downsampling_number
        self.mask_net = None
        self.show_every = show_every
        self.bg_hint = bg_hint
        self.fg_hint = fg_hint
        self.left_net = None
        self.right_net = None
        self.images = None
        self.images_torch = None
        self.left_net_inputs = None
        self.right_net_inputs = None
        self.mask_net_inputs = None
        self.left_net_outputs = None
        self.right_net_outputs = None
        self.second_step_done = False
        self.mask_net_outputs = None
        self.parameters = None
        self.gngc_loss = None
        self.fixed_masks = None
        self.blur_function = None
        self.first_step_iter_num = first_step_iter_num
        self.second_step_iter_num = second_step_iter_num
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.gngc = None
        self.blur = None
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        left_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.left_net = left_net.type(torch.cuda.FloatTensor)

        right_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.right_net = right_net.type(torch.cuda.FloatTensor)

        mask_net = skip_mask(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(torch.cuda.FloatTensor)

    def _init_images(self):
        self.images = get_imresize_downsampled(self.image, downsampling_factor=self.downsampling_factor,
                                               downsampling_number=self.downsampling_number)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]
        if self.bg_hint is not None:
            assert self.bg_hint.shape[1:] == self.image.shape[1:], (self.bg_hint.shape[1:], self.image.shape[1:])
            self.bg_hints = get_imresize_downsampled(self.bg_hint, downsampling_factor=self.downsampling_factor,
                                                     downsampling_number=self.downsampling_number)
            self.bg_hints_torch = [np_to_torch(bg_hint).type(torch.cuda.FloatTensor) for bg_hint in self.bg_hints]
        else:
            self.bg_hints = None
        if self.fg_hint is not None:
            assert self.fg_hint.shape[1:] == self.image.shape[1:]
            self.fg_hints = get_imresize_downsampled(self.fg_hint, downsampling_factor=self.downsampling_factor,
                                                     downsampling_number=self.downsampling_number)
            self.fg_hints_torch = [np_to_torch(fg_hint).type(torch.cuda.FloatTensor) for fg_hint in self.fg_hints]
        else:
            self.fg_hints = None

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        self.left_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).type(torch.cuda.FloatTensor).detach()
                                for image in self.images_torch]
        self.right_net_inputs = self.left_net_inputs
        input_type = 'noise'
        self.mask_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).type(torch.cuda.FloatTensor).detach()
                                for image in self.images_torch]

    def _init_parameters(self):
        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.right_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.gngc_loss = YIQGNGCLoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)
        self.blur_function = StdLoss().type(data_type)
        self.gradient_loss = GradientLoss().type(data_type)
        self.gray_loss = GrayLoss().type(data_type)

    def _init_all(self):
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        self._init_noise()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # step 1
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.first_step_iter_num):
            optimizer.zero_grad()
            self._step1_optimization_closure(j)
            self._finalize_iteration()
            if self.plot_during_training:
                self._iteration_plot_closure(j)
            optimizer.step()
        self._update_result_closure()
        if self.plot_during_training:
            self._step_plot_closure(1)
        # self.finalize_first_step()
        # step 2
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.second_step_iter_num):
            optimizer.zero_grad()
            self._step2_optimization_closure(j)
            self._finalize_iteration()
            if self.second_step_done:
                break
            if self.plot_during_training:
                self._iteration_plot_closure(j)
            optimizer.step()
        self._update_result_closure()
        if self.plot_during_training:
            self._step_plot_closure(2)

    def finalize_first_step(self):
        left = torch_to_np(self.left_net_outputs[0])
        right = torch_to_np(self.right_net_outputs[0])
        save_image(self.image_name + "_1_left", left)
        save_image(self.image_name + "_1_right", right)
        save_image(self.image_name + "_hint1", self.bg_hint)
        save_image(self.image_name + "_hint2", self.fg_hint)
        save_image(self.image_name + "_hint1_masked", self.bg_hint * self.image)
        save_image(self.image_name + "_hint2_masked", self.fg_hint * self.image)

    def finalize(self):
        save_image(self.image_name + "_left", self.best_result.left)
        save_image(self.image_name + "_learned_mask", self.best_result.learned_mask)
        save_image(self.image_name + "_right", self.best_result.right)
        save_image(self.image_name + "_original", self.images[0])
        # save_image(self.image_name + "_fg_bg", ((self.fg_hint - self.bg_hint) + 1) / 2)
        save_image(self.image_name + "_mask", self.best_result.mask)

    def _update_result_closure(self):
        self._finalize_iteration()
        self._fix_mask()
        self.current_result = SegmentationResult(mask=self.fixed_masks[0],
                                                 left=torch_to_np(self.left_net_outputs[0]),
                                                 right=torch_to_np(self.right_net_outputs[0]),
                                                 learned_mask=torch_to_np(self.mask_net_outputs[0]),
                                                 psnr=self.current_psnr)
        if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
            self.best_result = self.current_result

    def _fix_mask(self):
        """
        fixing the masks using soft matting
        :return:
        """
        masks_np = [torch_to_np(mask) for mask in self.mask_net_outputs]
        new_mask_nps = [np.array([guidedFilter(image_np.transpose(1, 2, 0).astype(np.float32),
                                               mask_np[0].astype(np.float32), 50, 1e-4)])
                        for image_np, mask_np in zip(self.images, masks_np)]

        def to_bin(x):
            v = np.zeros_like(x)
            v[x > 0.5] = 1
            return v

        self.fixed_masks = [to_bin(m) for m in new_mask_nps]

    def _initialize_step1(self, iteration):
        self._initialize_any_step(iteration)

    def _initialize_step2(self, iteration):
        self._initialize_any_step(iteration)

    def _initialize_any_step(self, iteration):
        if iteration == self.second_step_iter_num - 1:
            reg_noise_std = 0
        elif iteration < 1000:
            reg_noise_std = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std = 1 / 1000.
        right_net_inputs = []
        left_net_inputs = []
        mask_net_inputs = []
        # creates left_net_inputs and right_net_inputs by adding small noise
        for left_net_original_input, right_net_original_input, mask_net_original_input \
                in zip(self.left_net_inputs, self.right_net_inputs, self.mask_net_inputs):
            left_net_inputs.append(
                left_net_original_input + (left_net_original_input.clone().normal_() * reg_noise_std))
            right_net_inputs.append(
                right_net_original_input + (right_net_original_input.clone().normal_() * reg_noise_std))
            mask_net_inputs.append(
                mask_net_original_input + (mask_net_original_input.clone().normal_() * reg_noise_std))
        # applies the nets
        self.left_net_outputs = [self.left_net(left_net_input) for left_net_input in left_net_inputs]
        self.right_net_outputs = [self.right_net(right_net_input) for right_net_input in right_net_inputs]
        self.mask_net_outputs = [self.mask_net(mask_net_input) for mask_net_input in mask_net_inputs]
        self.total_loss = 0
        self.gngc = 0
        self.blur = 0

    def _step1_optimization_closure(self, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_step1(iteration)
        if self.fg_hints is not None and self.bg_hints is not None:
            self._step1_optimize_with_hints(iteration)
        else:
            self._step1_optimize_without_hints(iteration)

    def _step2_optimization_closure(self, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_step2(iteration)
        if self.fg_hints is not None and self.bg_hints is not None:
            self._step2_optimize_with_hints(iteration)
        else:
            self._step2_optimize_without_hints(iteration)

    def _step1_optimize_without_hints(self, iteration):
        self.total_loss += sum(self.l1_loss(torch.ones_like(mask_net_output) / 2, mask_net_output) for
                               mask_net_output in self.mask_net_outputs)
        self.total_loss.backward(retain_graph=True)

    def _step1_optimize_with_hints(self, iteration):
        """
        optimization, where hints are given
        :param iteration:
        :return:
        """
        self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                               left_net_output, fg_hint, image_torch
                               in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
        self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                               right_net_output, bg_hint, image_torch
                               in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))

        self.total_loss += sum(self.l1_loss(((fg_hint - bg_hint) + 1) / 2, mask_net_output) for
                               fg_hint, bg_hint, mask_net_output in
                               zip(self.fg_hints_torch, self.bg_hints_torch, self.mask_net_outputs))
        self.total_loss.backward(retain_graph=True)

    def _step2_optimize_without_hints(self, iteration):
        for left_out, right_out, mask_out, original_image_torch in zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs,
                                                                       self.images_torch):
            self.total_loss += 0.5 * self.l1_loss(mask_out * left_out + (1 - mask_out) * right_out,
                                                  original_image_torch)
            self.current_gradient = self.gray_loss(mask_out)
            # self.current_gradient = self.gradient_loss(mask_out)
            self.total_loss += (0.01 * (iteration // 100)) * self.current_gradient
        self.total_loss.backward(retain_graph=True)

    def _step2_optimize_with_hints(self, iteration):
        if iteration <= 1000:
            self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                                   left_net_output, fg_hint, image_torch
                                   in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
            self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                                   right_net_output, bg_hint, image_torch
                                   in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))

        for left_out, right_out, mask_out, original_image_torch in zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs,
                                                                       self.images_torch):
            self.total_loss += 0.5 * self.l1_loss(mask_out * left_out + (1 - mask_out) * right_out,
                                                  original_image_torch)
            self.current_gradient = self.gray_loss(mask_out)
            # self.current_gradient = self.gradient_loss(mask_out)
            iteration = min(iteration, 1000)
            self.total_loss += (0.001 * (iteration // 100)) * self.current_gradient
        self.total_loss.backward(retain_graph=True)

    def _finalize_iteration(self):
        left_out_np = torch_to_np(self.left_net_outputs[0])
        right_out_np = torch_to_np(self.right_net_outputs[0])
        original_image = self.images[0]
        mask_out_np = torch_to_np(self.mask_net_outputs[0])
        self.current_psnr = compare_psnr(original_image, mask_out_np * left_out_np + (1 - mask_out_np) * right_out_np)
        # TODO: run only in the second step
        if self.current_psnr > 30:
            self.second_step_done = True

    def _iteration_plot_closure(self, iter_number):
        if self.current_gradient is not None:
            print('Iteration {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                                   self.current_gradient.item(),
                                                                                   self.current_psnr),
                  '\r', end='')
        else:
            print('Iteration {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                        self.current_psnr),
                  '\r', end='')
        if iter_number % self.show_every == self.show_every - 1:
            self._plot_with_name(iter_number)

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        self._plot_with_name("step_{}".format(step_number))

    def _plot_with_name(self, name):
        if self.fg_hint is not None and self.bg_hint is not None:
            plot_image_grid("left_right_hints_{}".format(name),
                            [np.clip(self.fg_hint, 0, 1),
                             np.clip(self.bg_hint, 0, 1)])
        for i, (left_out, right_out, mask_out, image) in enumerate(zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs, self.images)):
            plot_image_grid("left_right_{}_{}".format(name, i),
                            [np.clip(torch_to_np(left_out), 0, 1),
                             np.clip(torch_to_np(right_out), 0, 1)])
            mask_out_np = torch_to_np(mask_out)
            plot_image_grid("learned_mask_{}_{}".format(name, i),
                            [np.clip(mask_out_np, 0, 1), 1 - np.clip(mask_out_np, 0, 1)])

            plot_image_grid("learned_image_{}_{}".format(name, i),
                            [np.clip(mask_out_np * torch_to_np(left_out) + (1 - mask_out_np) * torch_to_np(right_out),
                                     0, 1), image])