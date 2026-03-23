import os
from decimal import Decimal

import torch
from tqdm import tqdm
import torch.nn.functional as F

import utility


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss

        # Optimizer and scheduler
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        # Resume checkpoint if specified
        self.start_epoch = 0
        if self.args.load != '.':
            optimizer_path = os.path.join(ckp.dir, 'optimizer.pt')
            if os.path.isfile(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path))
            # Use log length to resume epoch count
            self.start_epoch = len(ckp.log)

        self.current_epoch = self.start_epoch
        self.error_last = 1e8

    # ==========================================================
    # Training for one epoch
    # ==========================================================
    def train(self):
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr_img, hr_img, _, idx_scale) in enumerate(self.loader_train):
            lr_img, hr_img = self.prepare([lr_img, hr_img])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            # Forward
            output = self.model(lr_img, idx_scale)
            if isinstance(output, tuple):
                sr, div_loss = output
            else:
                sr = output
                div_loss = 0

            # Reconstruction loss
            if isinstance(sr, list):
                recon_loss = sum(self.loss(s, hr_img) for s in sr) / len(sr)
                sr = sr[-1]
            else:
                recon_loss = self.loss(sr, hr_img)

            # Total loss
            loss = recon_loss + div_loss

            # Backward with gradient clipping
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.scheduler.step()
            else:
                print(f'Skip batch {batch + 1}! Loss: {loss.item()}')

            timer_model.hold()

            if (batch + 1) * self.args.batch_size % self.args.print_every == 0:
                self.ckp.write_log(
                    '==> [{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()
                    )
                )

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if not self.args.cpu:
            torch.cuda.synchronize()

    # ==========================================================
    # Testing / Evaluation
    # ==========================================================
    def test(self):
        self.ckp.write_log(f'\nEvaluation at Epoch {self.current_epoch + 1}:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for lr_img, hr_img, filename, _ in tqdm_test:
                    filename = filename[0]
                    no_eval = (hr_img.nelement() == 1)

                    if not no_eval:
                        lr_img, hr_img = self.prepare([lr_img, hr_img])
                    else:
                        lr_img = self.prepare([lr_img])[0]

                    output = self.model(lr_img, idx_scale)
                    sr = output[0] if isinstance(output, tuple) else output
                    sr = sr[-1] if isinstance(sr, list) else sr
                    sr = utility.quantize(sr, self.args.rgb_range)

                    if not no_eval:
                        hr_img = F.interpolate(hr_img, size=sr.shape[2:], mode='bilinear', align_corners=False)
                        eval_acc += utility.calc_psnr(
                            sr,
                            hr_img,
                            scale,
                            self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        save_list = [sr]
                        if not no_eval:
                            save_list.extend([lr_img, hr_img])
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        if not self.args.test_only:
            self.ckp.save(self, self.current_epoch + 1, is_best=(best[1][0] + 1 == self.current_epoch + 1))

    # ==========================================================
    # Data Preparation
    # ==========================================================
    def prepare(self, tensors):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(t) for t in tensors]

    # ==========================================================
    # Termination / Epoch Control
    # ==========================================================
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            self.current_epoch += 1
            return self.current_epoch > self.args.epochs