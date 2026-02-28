import os
from importlib import import_module

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('models.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        if args.precision == 'half':
            self.model.half()

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # Safe load
        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

        if args.print_model:
            print(self.model)

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        target = self.get_model()

        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        # ---------------- Self Ensemble ----------------
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = lambda _x: self.forward_chop(_x)
            else:
                forward_function = lambda _x: self.model(_x, idx_scale)

            return self.forward_x8(x, forward_function)

        # ---------------- Chop Only ----------------
        elif self.chop and not self.training:
            return self.forward_chop(x)

        # ---------------- Normal Forward ----------------
        else:
            return self.model(x, idx_scale)

    # ==========================================================
    # Utility
    # ==========================================================
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    # ==========================================================
    # Save
    # ==========================================================
    def save(self, apath, epoch, is_best=False):
        target = self.get_model()

        model_dir = os.path.join(apath, 'model')
        os.makedirs(model_dir, exist_ok=True)

        torch.save(
            target.state_dict(),
            os.path.join(model_dir, 'model_latest.pt')
        )

        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(model_dir, 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(model_dir, f'model_{epoch}.pt')
            )

    # ==========================================================
    # Load
    # ==========================================================
    def load(self, apath, pre_train='.', resume=-1, cpu=False):

        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        model_dir = os.path.join(apath, 'model')
        latest_path = os.path.join(model_dir, 'model_latest.pt')

        # Resume latest
        if resume == -1:
            if os.path.isfile(latest_path):
                print('Resuming from latest checkpoint...')
                self.get_model().load_state_dict(
                    torch.load(latest_path, **kwargs),
                    strict=False
                )
            else:
                print('No checkpoint found. Training from scratch.')

        # Load pretrained model
        elif resume == 0:
            if pre_train != '.' and os.path.isfile(pre_train):
                print(f'Loading pretrained model from {pre_train}')
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
            else:
                print('No valid pretrained model found. Training from scratch.')

        # Resume specific epoch
        else:
            specific_path = os.path.join(model_dir, f'model_{resume}.pt')
            if os.path.isfile(specific_path):
                print(f'Resuming from checkpoint {resume}...')
                self.get_model().load_state_dict(
                    torch.load(specific_path, **kwargs),
                    strict=False
                )
            else:
                print('Specified checkpoint not found. Training from scratch.')

    # ==========================================================
    # Forward Chop
    # ==========================================================
    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()

        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]
        ]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch, self.idx_scale)

                if isinstance(sr_batch, tuple):
                    sr_batch = sr_batch[0]

                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size)
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output