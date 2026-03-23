import os
import imageio
import numpy as np
import torch.utils.data as data

import mydata.common as common

class SRData(data.Dataset):
    """SR dataset interface: supports training, validation, and benchmark datasets with filtering of missing files."""

    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        print("Initializing SRData with args.scale:", self.scale)
        self._set_filesystem(args.dir_data)

        def _load_bin():
            """Load preprocessed binary files for HR and LR images."""
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [np.load(self._name_lrbin(s)) for s in self.scale]
            print("Loaded binary HR and LR files.")

        # Load dataset
        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan_filtered()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan_filtered()
            if args.ext.find('reset') >= 0:
                print('Preparing separated binary files...')
                for v in self.images_hr:
                    hr = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = imageio.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [v.replace(self.ext, '.npy') for v in self.images_hr]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]
        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading binary file...')
                _load_bin()
            except:
                print('Preparing binary file...')
                bin_path = os.path.join(self.apath, 'bin')
                os.makedirs(bin_path, exist_ok=True)
                list_hr, list_lr = self._scan_filtered()
                hr = [imageio.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [imageio.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                print('Loading binary file after preparation...')
                _load_bin()
        else:
            raise ValueError('Please define a valid data type for args.ext.')

    def _set_filesystem(self, dir_data):
        """Set dataset folders for training, validation, or benchmark datasets."""
        self.apath = dir_data

        if self.benchmark:
            # Benchmark dataset paths
            self.dir_hr = os.path.join(self.apath, self.args.data_test, 'HR', f'X{self.scale[0]}')
            self.dir_lr = [os.path.join(self.apath, self.args.data_test, 'LR_bicubic', f'X{s}') for s in self.scale]
            print("Benchmark HR folder:", self.dir_hr)
            print("Benchmark LR folders:", self.dir_lr)
        else:
            # Training/validation dataset paths
            if self.train:
                self.dir_hr = os.path.join(self.apath, 'DIV2K', 'DIV2K_train_HR')
                self.dir_lr = [os.path.join(self.apath, 'DIV2K', f'DIV2K_train_LR_bicubic/X{s}') for s in self.scale]
            else:
                self.dir_hr = os.path.join(self.apath, 'DIV2K', 'DIV2K_valid_HR')
                self.dir_lr = [os.path.join(self.apath, 'DIV2K', f'DIV2K_valid_LR_bicubic/X{s}') for s in self.scale]
            print("HR folder:", self.dir_hr)
            print("LR folders:", self.dir_lr)

    def _scan(self):
        """Scan HR and LR files and match LR filenames to HR filenames."""
        hr_list = sorted([os.path.join(self.dir_hr, f) for f in os.listdir(self.dir_hr) if f.endswith('.png')])
        if len(hr_list) == 0:
            print("WARNING: No HR images found in", self.dir_hr)

        lr_list = []
        for s in range(len(self.scale)):
            scale = self.scale[s]
            lr_folder = self.dir_lr[s]
            lr_files = []

            for hr_path in hr_list:
                hr_name = os.path.splitext(os.path.basename(hr_path))[0]
                lr_name = f"{hr_name}x{scale}.png"
                lr_path = os.path.join(lr_folder, lr_name)
                if os.path.exists(lr_path):
                    lr_files.append(lr_path)
                else:
                    print(f"Missing LR image: {lr_path}")

            lr_list.append(lr_files)

        # Ensure HR and LR counts match
        min_len = min(len(hr_list), min(len(lr) for lr in lr_list))
        hr_list = hr_list[:min_len]
        for s in range(len(lr_list)):
            lr_list[s] = lr_list[s][:min_len]

        print(f"Found {len(hr_list)} HR images and {len(lr_list[0])} LR images for scale {self.scale[0]}")
        return hr_list, lr_list

    def _scan_filtered(self):
        """Scan dataset and filter missing files."""
        hr_list, lr_list = self._scan()
        hr_list = [f for f in hr_list if os.path.exists(f)]
        filtered_lr_list = []
        for s in range(len(self.scale)):
            lr_filtered = [f for f in lr_list[s] if os.path.exists(f)]
            filtered_lr_list.append(lr_filtered)
        print(f"After filtering: {len(hr_list)} HR images, {len(filtered_lr_list[0])} LR images")
        return hr_list, filtered_lr_list

    def _name_hrbin(self):
        return os.path.join(self.apath, 'bin', f'{self.split}_HR.npy')

    def _name_lrbin(self, scale):
        return os.path.join(self.apath, 'bin', f'{self.split}_LR_X{scale}.npy')

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        try:
            lr = self.images_lr[self.idx_scale][idx]
            hr = self.images_hr[idx]
        except IndexError:
            print(f"IndexError: idx {idx}, HR length {len(self.images_hr)}, LR length {len(self.images_lr[self.idx_scale])}")
            raise

        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)
        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(lr, hr, patch_size, scale)
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
        print(f"Set idx_scale to {self.idx_scale}")