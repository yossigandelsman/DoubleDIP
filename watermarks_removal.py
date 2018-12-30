from collections import namedtuple

from skimage.measure import compare_psnr

from net import *
from net.downsampler import *
from net.losses import StdLoss, GradientLoss, ExtendedL1Loss, GrayLoss
from net.losses import ExclusionLoss
from net.noise import get_noise

WatermarkResult = namedtuple("WatermarkResult", ['clean', 'watermark', 'mask', 'psnr'])


class Watermark(object):
    def __init__(self, image_name, image, plot_during_training=True, num_iter_first_step=4000,
                 num_iter_second_step=7000,
                 watermark_hint=None):
        self.image = image
        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.watermark_hint_torchs = None
        self.watermark_hint = watermark_hint
        self.clean_net = None
        self.watermark_net = None
        self.image_torchs = None
        self.clean_net_inputs = None
        self.watermark_net_inputs = None
        self.clean_net_output = None
        self.watermark_net_output = None
        self.parameters = None
        self.blur_function = None
        self.num_iter_first_step = num_iter_first_step  # per step
        self.num_iter_second_step = num_iter_second_step  # per step
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.blur = None
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        clean = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.clean_net = clean.type(torch.cuda.FloatTensor)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.type(torch.cuda.FloatTensor)

        mask = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.mask_net = mask.type(torch.cuda.FloatTensor)

    def _init_images(self):
        image_aug = create_augmentations(self.image)
        self.image_torchs = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in image_aug]
        water_mark_aug = create_augmentations(self.watermark_hint)
        self.watermark_hint_torchs = [np_to_torch(watr).type(torch.cuda.FloatTensor) for watr in water_mark_aug]

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        clean_net_inputs = create_augmentations(torch_to_np(get_noise(self.input_depth, input_type,
                                                                      (self.image_torchs[0].shape[2],
                                                                       self.image_torchs[0].shape[3])).type(torch.cuda.FloatTensor).detach()))
        self.clean_net_inputs = [np_to_torch(clean_net_input).type(torch.cuda.FloatTensor).detach()
                                 for clean_net_input in clean_net_inputs]

        watermark_net_inputs = create_augmentations(torch_to_np(get_noise(self.input_depth, input_type,
                                                                          (self.image_torchs[0].shape[2],
                                                                       self.image_torchs[0].shape[3])).type(
            torch.cuda.FloatTensor).detach()))
        self.watermark_net_inputs = [np_to_torch(clean_net_input).type(torch.cuda.FloatTensor).detach()
                                 for clean_net_input in watermark_net_inputs]

        mask_net_inputs = create_augmentations(torch_to_np(get_noise(self.input_depth, input_type,
                                                                     (self.image_torchs[0].shape[2],
                                                                       self.image_torchs[0].shape[3])).type(
            torch.cuda.FloatTensor).detach()))
        self.mask_net_inputs = [np_to_torch(clean_net_input).type(torch.cuda.FloatTensor).detach()
                                 for clean_net_input in mask_net_inputs]

    def _init_parameters(self):
        self.parameters = [p for p in self.clean_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)

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
        self._step_initialization_closure(0)
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter_first_step):
            optimizer.zero_grad()
            self._step1_optimization_closure(j, 0)
            # if self.plot_during_training:
            #     self._iteration_plot_closure(0, j)
            optimizer.step()
        #self._update_result_closure(0)
        # self._step_plot_closure(0)
        # step 2
        self._step_initialization_closure(1)
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter_second_step):
            optimizer.zero_grad()
            self._step2_optimization_closure(j, 1)
            if self.plot_during_training:
                self._iteration_plot_closure(1, j)
            optimizer.step()
        self._update_result_closure(1)
        self._step_plot_closure(1)

    def finalize(self):
        save_image(self.image_name + "_watermark", self.best_result.watermark)
        save_image(self.image_name + "_clean", self.best_result.clean)
        save_image(self.image_name + "_original", self.image)
        save_image(self.image_name + "_mask", self.best_result.mask)
        save_image(self.image_name + "_final", (1 - self.watermark_hint) * self.image +
                   self.best_result.clean * self.watermark_hint)

    def _update_result_closure(self, step):
        self.current_result = WatermarkResult(clean=torch_to_np(self.clean_net_output),
                                              watermark=torch_to_np(self.watermark_net_output),
                                              mask=torch_to_np(self.mask_net_output),
                                              psnr=self.current_psnr)
        # if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
        self.best_result = self.current_result

    def _step_initialization_closure(self, step):
        """
        at each start of step, we apply this
        :param step:
        :return:
        """
        # we updating the inputs to new noises
        # self._init_nets()
        # self._init_parameters()
        # self._init_noise()
        pass

    def _get_augmentation(self, iteration):
        if iteration % 4 in [1, 2, 3]:
            return 0
        iteration //= 2
        return iteration % 8

    def _step2_optimization_closure(self, iteration, step):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        if iteration == self.num_iter_second_step - 1:
            reg_noise_std = 0
        else:
            reg_noise_std = (1 / 1000.) * (iteration // 700)

        aug = self._get_augmentation(iteration)
        if iteration == self.num_iter_second_step - 1:
            aug  = 0
        # creates left_net_inputs and right_net_inputs by adding small noise
        clean_net_input = self.clean_net_inputs[aug] + (self.clean_net_inputs[aug].clone().normal_() * reg_noise_std)
        watermark_net_input = self.watermark_net_inputs[aug] + (self.watermark_net_inputs[aug].clone().normal_() * reg_noise_std)
        mask_net_input = self.mask_net_inputs[aug]  # + (self.mask_net_input.clone().normal_() * reg_noise_std)
        # applies the nets
        self.clean_net_output = self.clean_net(clean_net_input)
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0
        # loss on clean region
        self.total_loss += self.extended_l1_loss(self.clean_net_output,
                                                 self.image_torchs[aug],
                                                 (1 - self.watermark_hint_torchs[aug]))
        # loss in second region
        self.total_loss += 0.5 * self.l1_loss(self.watermark_hint_torchs[aug] *
                                              self.mask_net_output * self.watermark_net_output
                                              +
                                              (1 - self.mask_net_output) * self.clean_net_output,
                                              self.image_torchs[aug])  # this part learns the watermark
        self.total_loss.backward(retain_graph=True)

    def _step1_optimization_closure(self, iteration, step):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        if iteration == self.num_iter_first_step - 1:
            reg_noise_std = 0
        else:
            reg_noise_std = (1 / 1000.) * (iteration // 300)  # TODO: make it dependant in the max number of iterations
        aug = self._get_augmentation(iteration)
        if iteration == self.num_iter_first_step - 1:
            aug = 0
        # creates left_net_inputs and right_net_inputs by adding small noise
        clean_net_input = self.clean_net_inputs[aug] + (self.clean_net_inputs[aug].clone().normal_() * reg_noise_std)
        # watermark_net_input = self.watermark_net_inputs[aug] #  + (self.watermark_net_input.clone().normal_())
        # mask_net_input = self.mask_net_inputs[aug]
        # applies the nets
        self.clean_net_output = self.clean_net(clean_net_input)
        self.total_loss = 0
        self.blur = 0
        self.total_loss += self.extended_l1_loss(self.clean_net_output,
                                                 self.image_torchs[aug],
                                                 (1 - self.watermark_hint_torchs[aug]))
        self.total_loss.backward(retain_graph=True)

    def _iteration_plot_closure(self, step_number, iter_number):
        if iter_number % 32 == 0:
            clean_out_np = torch_to_np(self.clean_net_output)
            watermark_out_np = torch_to_np(self.watermark_net_output)
            mask_out_np = torch_to_np(self.watermark_net_output)
            if step_number == 0:
                self.current_psnr = 0
            self.current_psnr = compare_psnr(self.image, mask_out_np * self.watermark_hint * watermark_out_np +
                                             (1 - mask_out_np) * clean_out_np)
            if self.current_gradient is not None:
                print('Iteration {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                                       self.current_gradient.item(),
                                                                                       self.current_psnr),
                      '\r', end='')
            else:
                print('Iteration {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                            self.current_psnr),
                      '\r', end='')

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        if self.watermark_hint is not None:
            plot_image_grid("watermark_hint_and_mask_{}".format(step_number),
                            [np.clip(self.watermark_hint, 0, 1),
                             np.clip(torch_to_np(self.mask_net_output), 0, 1)])

        plot_image_grid("watermark_clean_{}".format(step_number),
                        [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
                         np.clip(torch_to_np(self.clean_net_output), 0, 1)])

        plot_image_grid("learned_image_{}".format(step_number),
                        [np.clip(self.watermark_hint * torch_to_np(self.watermark_net_output) +
                                 torch_to_np(self.clean_net_output),
                                 0, 1), self.image])


def remove_watermark(image_name, image, fg):
    results = []
    for i in range(3):
        s = Watermark(image_name+"_{}".format(i), image, watermark_hint=fg)
        s.optimize()
        s.finalize()
        results.append(s.best_result)

    save_image(image_name + "_watermark", median([best_result.watermark for best_result in results]))
    save_image(image_name + "_clean", median([best_result.clean for best_result in results]))
    save_image(image_name + "_original", image)
    save_image(image_name + "_final", (1 - fg) * image + fg * median([best_result.clean for best_result in results]))
    save_image(image_name + "_mask", median([best_result.mask for best_result in results]))
    save_image(image_name + "_hint", fg)
    recovered_mask = fg * median([best_result.mask for best_result in results])
    clear_image_places = np.zeros_like(recovered_mask)
    clear_image_places[recovered_mask < 0.1] = 1
    save_image(image_name + "_real_final", clear_image_places * image + (1 - clear_image_places) *
               median([best_result.clean for best_result in results]))
    recovered_watermark = fg * median([best_result.watermark * best_result.mask for best_result in results])
    save_image(image_name + "_recovered_watermark", recovered_watermark)


ManyImageWatermarkResult = namedtuple("ManyImageWatermarkResult", ['cleans', 'mask', 'watermark', 'psnr'])


class ManyImagesWatermarkNoHint(object):
    def __init__(self, images_names, images, plot_during_training=True, num_iter_per_step=4000, num_step=1):
        self.images = images
        self.images_names = images_names
        self.plot_during_training = plot_during_training
        self.clean_nets = []
        self.watermark_net = None
        self.steps = num_step
        self.images_torch = None
        self.clean_nets_inputs = None
        self.clean_nets_outputs = None
        self.watermark_net_input = None
        self.watermark_net_output = None
        self.mask_net_input = None
        self.mask_net_output = None
        self.parameters = None
        self.blur_function = None
        self.num_iter_per_step = num_iter_per_step  # per step
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.blur = None
        self.current_psnr = 0
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        cleans = [skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU') for _ in self.images]

        self.clean_nets = [clean.type(torch.cuda.FloatTensor) for clean in cleans]

        mask_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(torch.cuda.FloatTensor)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.type(torch.cuda.FloatTensor)

    def _init_images(self):
        # convention - first dim is all the images, second dim is the augmenations
        self.images_torch = [[np_to_torch(aug).type(torch.cuda.FloatTensor)
                              for aug in create_augmentations(image)] for image in self.images]

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        self.clean_nets_inputs = []
        for image_idx in range(len(self.images)):
            original_noise = get_noise(self.input_depth, input_type,
                                                (self.images_torch[image_idx][0].shape[2],
                                                 self.images_torch[image_idx][0].shape[3])).type(torch.cuda.FloatTensor).detach()
            augmentations = create_augmentations(torch_to_np(original_noise))
            self.clean_nets_inputs.append([np_to_torch(aug).type(torch.cuda.FloatTensor).detach() for aug in augmentations])

        original_noise = get_noise(self.input_depth, input_type,
                                  (self.images_torch[0][0].shape[2],
                                   self.images_torch[0][0].shape[3])).type(torch.cuda.FloatTensor).detach()
        augmentations = create_augmentations(torch_to_np(original_noise))
        self.mask_net_input = [np_to_torch(aug).type(torch.cuda.FloatTensor).detach() for aug in augmentations]

        original_noise = get_noise(self.input_depth, input_type,
                                   (self.images_torch[0][0].shape[2],
                                    self.images_torch[0][0].shape[3])).type(torch.cuda.FloatTensor).detach()
        augmentations = create_augmentations(torch_to_np(original_noise))
        self.watermark_net_input = [np_to_torch(aug).type(torch.cuda.FloatTensor).detach() for aug in augmentations]


    def _init_parameters(self):
        self.parameters = sum([[p for p in clean_net.parameters()] for clean_net in self.clean_nets], []) + \
                          [p for p in self.mask_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
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
        for step in range(self.steps):
            self._step_initialization_closure(step)
            optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
            for j in range(self.num_iter_per_step):
                optimizer.zero_grad()
                self._optimization_closure(j, step)
                if self.plot_during_training:
                    self._iteration_plot_closure(j, step)
                optimizer.step()
            self._update_result_closure(step)
            # self._step_plot_closure(step)

    def finalize(self):
        for image_name, clean, image in zip(self.images_names, self.best_result.cleans, self.images):
            save_image(image_name + "_watermark", self.best_result.watermark)
            save_image(image_name + "_mask", self.best_result.mask)
            save_image(image_name + "_obtained_mask", self.best_result.mask * self.best_result.watermark)
            save_image(image_name + "_clean", clean)
            save_image(image_name + "_original", image)

    def _update_result_closure(self, step):
        self.current_result = ManyImageWatermarkResult(cleans=[torch_to_np(c) for c in self.clean_nets_outputs],
                                                       watermark=torch_to_np(self.watermark_net_output),
                                                       mask=torch_to_np(self.mask_net_output),
                                                       psnr=self.current_psnr)
        if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
            self.best_result = self.current_result

    def _step_initialization_closure(self, step):
        """
        at each start of step, we apply this
        :param step:
        :return:
        """
        # we updating the inputs to new noises
        # self._init_nets()
        # self._init_parameters()
        # self._init_noise()
        pass

    def _get_augmentation(self, iteration):
        if iteration % 4 in [1, 2, 3]:
            return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, iteration, step):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        aug = self._get_augmentation(iteration)
        if iteration == self.num_iter_per_step - 1:
            reg_noise_std = 0
            aug = 0
        else:
            reg_noise_std = (1 / 1000.) * (iteration // 400)
        # creates left_net_inputs and right_net_inputs by adding small noise
        clean_nets_inputs = [clean_net_input[aug] + (clean_net_input[aug].clone().normal_() * reg_noise_std)
                             for clean_net_input in self.clean_nets_inputs]
        watermark_net_input = self.watermark_net_input[aug] # + (self.watermark_net_input[aug].clone().normal_() * reg_noise_std)
        mask_net_input = self.mask_net_input[aug]
        # applies the nets
        self.clean_nets_outputs = [clean_net(clean_net_input) for clean_net, clean_net_input
                                   in zip(self.clean_nets, clean_nets_inputs)]
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0
        self.blur = 0

        self.total_loss += sum(self.l1_loss(self.watermark_net_output * self.mask_net_output +
                                            clean_net_output * (1 - self.mask_net_output), image_torch[aug])
                               for clean_net_output, image_torch in zip(self.clean_nets_outputs, self.images_torch))
        self.total_loss.backward(retain_graph=True)

    def _iteration_plot_closure(self, iteration, step):
        if iteration % 32 == 0:
            clean_out_nps = [torch_to_np(clean_net_output) for clean_net_output in self.clean_nets_outputs]
            watermark_out_np = torch_to_np(self.watermark_net_output)
            mask_out_np = torch_to_np(self.mask_net_output)
            self.current_psnr = compare_psnr(self.images[0], clean_out_nps[0] * (1 - mask_out_np) +
                                             mask_out_np * watermark_out_np)
            print('Iteration {:5d} PSNR {:5f} '.format(iteration, self.current_psnr),
                      '\r', end='')

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        for image_name, image, clean_net_output in zip(self.images_names, self.images, self.clean_nets_outputs):
            plot_image_grid(image_name + "_watermark_clean_{}".format(step_number),
                            [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
                             np.clip(torch_to_np(clean_net_output), 0, 1)])
            plot_image_grid(image_name + "_learned_image_{}".format(step_number),
                            [np.clip(torch_to_np(self.watermark_net_output) * torch_to_np(self.mask_net_output) +
                                     (1 - torch_to_np(self.mask_net_output)) * torch_to_np(clean_net_output),
                                     0, 1), image])


def remove_watermark_many_images(imgs_names, imgs, final_name, iters=3):
    results = []
    for img_name, original in zip(imgs_names, imgs):
        save_image(final_name + "_{}_original".format(img_name), original)
    for i in range(iters):
        s = ManyImagesWatermarkNoHint([name + "_{}".format(i) for name in imgs_names], imgs, plot_during_training=False)
        s.optimize()
        s.finalize()
        results.append(s.best_result)
    obtained_watermark = median([result.mask * result.watermark for result in results])

    obtained_imgs = [median([result.cleans[i] for result in results]) for i in range(len(imgs))]

    v = np.zeros_like(obtained_watermark)
    v[obtained_watermark < 0.1] = 1
    final_imgs = []
    for im, obt_im in zip(imgs, obtained_imgs):
        final_imgs.append(v * im + (1 - v) * obt_im)
    for img_name, final in zip(imgs_names, final_imgs):
        save_image(final_name + "_{}_final".format(img_name), final)
    obtained_watermark[obtained_watermark < 0.1] = 0
    save_image(final_name + "_final_watermark", obtained_watermark)


if __name__ == "__main__":
    # with many images:
    im1 = prepare_image('images/fotolia1.jpg')
    im2 = prepare_image('images/fotolia2.jpg')
    im3 = prepare_image('images/fotolia3.jpg')
    remove_watermark_many_images(['f1', 'f2', 'f3'], [im1, im2, im3], "fotolia_many_images")
    # with one image and bounding box:
    im = prepare_image('images/fotolia.jpg')
    fg = prepare_image('images/fotolia_watermark.png')
    remove_watermark("fotolia_one_image", im, fg)