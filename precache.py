import numpy as np, h5py
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm
from config import ProjectConfig

H5 = Path(ProjectConfig.DATA_PATH)
OUT = H5.parent / "images_224_uint8.npy"

if OUT.is_file():
    print(f"already cached: {OUT}")
else:
    with h5py.File(H5, "r") as f:
        N = f["images"].shape[0]
        imgs = np.empty((N, 224, 224, 3), dtype=np.uint8)
        for i in tqdm(range(N), desc="resizing"):
            img = f["images"][i]
            imgs[i] = (resize(img, (224,224), preserve_range=True,
                              anti_aliasing=True)).clip(0,255).astype(np.uint8)
    np.save(OUT, imgs)
    print(f"saved {OUT} ({imgs.nbytes/1e9:.2f} GB)")