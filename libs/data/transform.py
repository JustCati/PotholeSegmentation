import torch
import numpy as np
from torchvision.transforms import v2 as T



class GaussianBlur(object):
    def __init__(self, p = 0.5, kernel_size = 5, sigma = 1):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            if not isinstance(img, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(img)}")

            dtype = img.dtype
            if not img.is_floating_point():
                img = img.to(torch.float32)

            kernel = T.GaussianBlur(self.kernel_size, sigma = self.sigma)
            out = kernel(img)
            if out.dtype != dtype:
                out = out.to(dtype)
            return out, target
        return img, target

    def __repr__(self):
        return self.__class__.__name__



class GaussianNoise(object):
    def __init__(self, p = 0.5, noise_p = 0.2, mean = 0, sigma = 50):
        self.p = p
        self.sigma = sigma
        self.mean = mean
        self.noise_p = noise_p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            if not isinstance(img, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(img)}")

            dtype = img.dtype
            if not img.is_floating_point():
                img = img.to(torch.float32)

            mask = torch.normal(mean = self.mean, std = self.sigma, size = img.shape)
            random_index = np.random.choice([0, 1], size = img.shape, p = [1 - self.noise_p, self.noise_p])
            mask = mask * random_index

            out = img + mask
            if out.dtype != dtype:
                out = out.to(dtype)
            return out, target
        return img, target

    def __repr__(self):
        return self.__class__.__name__
