import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.transform import resize

from config import ProjectConfig


class Galaxy10Dataset(Dataset):
    """
    Feeds (image_tensor, label) from Galaxy10_DECals.h5.

    Parameters
    ----------
    h5_path     : str or Path
    indices     : array of row indices into the HDF5 (train/val/test split)
    image_size  : int, target H=W
    augment     : bool, apply random flips/rotations/colour-jitter (train only)
    """

    def __init__(self, h5_path, indices, image_size, augment=False):
        self.h5_path    = str(h5_path)
        self.indices    = np.asarray(indices, dtype=np.int64)
        self.image_size = image_size
        self.augment    = augment

        self._file = None   # lazy-open per worker

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=30,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        else:
            self.transform = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        f   = self._open()
        row = int(self.indices[i])
        img = f["images"][row]           # (256, 256, 3) uint8
        label = int(f["ans"][row])

        img = resize(
            img,
            (self.image_size, self.image_size),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) / 255.0

        tensor = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass