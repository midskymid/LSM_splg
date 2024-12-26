import numpy as np
import onnxruntime as ort
import cv2
import torch
import os
from typing import List, Optional, Union, Tuple
from pathlib import Path
import kornia
from types import SimpleNamespace

def print_gpu_usage():
    """To display GPU usage when the program runs to a specified position"""
    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # transfer to GB
    gpu_memory_cached = torch.cuda.memory_reserved() / 1024 ** 3
    
    print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
    print(f"GPU Memory Cached: {gpu_memory_cached:.2f} GB")

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()}

def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f'No image at path {path}.')
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image

def resize_image(image: np.ndarray, size: Union[List[int], int],
                 fn: str = 'max', interp: Optional[str] = 'area',
                 ) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {'max': max, 'min': min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.tensor(image / 255., dtype=torch.float)

def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)

class ImagePreprocessor:
    default_conf = {
        'resize': None,  # target edge length, None for no resizing
        'side': 'long',
        'interpolation': 'bilinear',
        'align_corners': None,
        'antialias': True,
        'grayscale': False,  # convert rgb to grayscale
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img, self.conf.resize, side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners)
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        if self.conf.grayscale and img.shape[-3] == 3:
            img = kornia.color.rgb_to_grayscale(img)
        elif not self.conf.grayscale and img.shape[-3] == 1:
            img = kornia.color.grayscale_to_rgb(img)
        return img, scale



