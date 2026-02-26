import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class MSDataLoader(DataLoader):
    def __init__(
        self,
        args,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=default_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(MSDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

        self.scale = args.scale
        self.train = dataset.train if hasattr(dataset, "train") else False

    def __iter__(self):
        for batch in super().__iter__():
            # Handle multi-scale logic here instead of worker internals
            idx_scale = 0
            if self.train and isinstance(self.scale, (list, tuple)) and len(self.scale) > 1:
                idx_scale = random.randrange(0, len(self.scale))
                if hasattr(self.dataset, "set_scale"):
                    self.dataset.set_scale(idx_scale)

            # Original code expected idx_scale appended
            if isinstance(batch, (list, tuple)):
                batch = list(batch)
                batch.append(idx_scale)
                yield batch
            else:
                yield batch
