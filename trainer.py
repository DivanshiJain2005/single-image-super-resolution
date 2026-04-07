import os
from decimal import Decimal

import torch
from tqdm import tqdm

import utility
import time
import torch.nn.functional as F

class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale  # 缩放尺度

        self.ckp = ckp  # checkpoint

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test  # now a list of loaders
        # model属性存储之前Model通过make_model实例化的EGDUN对象
        self.model = my_model
        # loss属性存储损失函数的实例化对象
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)  # 优化器
        self.scheduler = utility.make_scheduler(args, self.optimizer)  # 调度器

        if self.args.load != '.':  # 不等于表示需要加载优化器的状态字典
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(ckp.dir, 'optimizer.pt')
                )
            )
            for _ in range(len(ckp.log)):  # 根据日志，也就是训练过程中的步数进行学习率调整
                self.scheduler.step()

        self.error_last = 1e8

    def test(self):
        epoch = self.scheduler.last_epoch + 2
        self.ckp.write_log('Evaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        is_best = False
        with torch.no_grad():
            for loader_test in self.loader_test:
                dataset_name = loader_test.dataset.dataset_name
                for idx_scale, scale in enumerate(self.scale):
                    eval_acc = 0
                    eval_ssim = 0
                    loader_test.dataset.set_scale(idx_scale)
                    tqdm_test = tqdm(loader_test, ncols=80)
                    for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                        filename = filename[0]
                        no_eval = (hr.nelement() == 1)
                        if not no_eval:
                            lr, hr = self.prepare([lr, hr])
                        else:
                            lr = self.prepare([lr])[0]
                        # Pad LR to avoid size mismatches in multi-stride blocks
                        pad_h = (16 - (lr.shape[-2] % 16)) % 16
                        pad_w = (16 - (lr.shape[-1] % 16)) % 16
                        if pad_h or pad_w:
                            lr = F.pad(lr, (0, pad_w, 0, pad_h), mode='reflect')

                        sr = self.model(lr, idx_scale)
                        if isinstance(sr, list):
                            sr = sr[-1]
                        sr = sr[:, :, :hr.shape[-2], :hr.shape[-1]]
                        sr = utility.quantize(sr, self.args.rgb_range)
                        sr_size = sr.shape[2:]
                        hr_size = hr.shape[2:]
                        if sr_size != hr_size:
                            sr = F.interpolate(sr, size=hr_size, mode='bilinear', align_corners=False)
                        save_list = [sr]
                        if not no_eval:
                            eval_acc += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range,
                                benchmark=loader_test.dataset.benchmark
                            )
                            eval_ssim += utility.calc_ssim(sr, hr, scale)
                            save_list.extend([lr, hr])
                        if self.args.save_results:
                            self.ckp.save_results(filename, save_list, scale, dataset_name)

                    n = len(loader_test)
                    self.ckp.log[-1, idx_scale] = eval_acc / n
                    best = self.ckp.log.max(0)
                    if best[1][idx_scale] + 1 == epoch:
                        is_best = True
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f}  SSIM: {:.4f} (Best PSNR: {:.3f} @epoch {})'.format(
                            dataset_name,
                            scale,
                            self.ckp.log[-1, idx_scale],
                            eval_ssim / n,
                            best[0][idx_scale],
                            best[1][idx_scale] + 1
                        )
                    )

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=is_best)

    def train(self):
        # torch.cuda.synchronize()
        # start = time.time()
        self.scheduler.step()  # 调整学习率
        self.loss.step()  # 优化损失函数
        epoch = self.scheduler.last_epoch + 2
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '\n[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        # torch.nn.module的内置方法，将切换模型的评估模式为训练模式
        self.model.train()
        # self.args.test_only = False
        # print("self.args.test_only_train:", self.args.test_only)
        timer_data, timer_model = utility.timer(), utility.timer()
        # tqdm_train = tqdm(self.loader_train, ncols=80)
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):

            lr, hr = self.prepare([lr, hr])  # 将传入的LR,HR图像转为半精度

            timer_data.hold()  # 暂停计时器
            timer_model.tic()  # 重启计时器

            self.optimizer.zero_grad()  # 清除梯度
            sr = self.model(lr, idx_scale)

            # 计算loss
            if isinstance(sr, list):  # 如果对应恢复的SR网络是一个列表格式，那么对应对每一个SR求损失，然后对损失求和然后求平均
                loss = 0
                for sr_ in sr:
                    loss += self.loss(sr_, hr)
                loss = loss / len(sr)
            else:
                loss = self.loss(sr, hr)

            if loss.item() < self.args.skip_threshold * self.error_last:  # 算出的损失比上一次还要小，可以进行更新
                loss.backward()  # 自动计算require_grad属性为真的张量的梯度
                self.optimizer.step()  # 更新参数
            else:  # 否则直接跳过这一个批次的数据
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()
            # batch编号从0开始，对应的+1表示编号从1开始，*batch_size用于计算当前已经训练的数据量
            if (batch + 1) * self.args.batch_size % self.args.print_every == 0:
                self.ckp.write_log('==> [{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),  # 模型运行时间
                    timer_data.release()))  # 数据加载时间

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))  # 结束批次训练日志记录
        self.error_last = self.loss.log[-1, -1]  # 最后又一批最后一个损失值，表示最终的训练损失值，用于后续的评估

        torch.cuda.synchronize()
        # end = time.time()
        # print("running time is", end - start)

    def prepare(self, l):  # 数据预处理，修改精度
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]  # 递归函数

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
