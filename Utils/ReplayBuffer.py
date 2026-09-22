import random
import torch


class ImagePool:
    def __init__(self, size=50):
        self.size = size
        self.images = []

    def query(self, batch):
        result = []
        for image in batch:
            image = image.detach().unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image)
                result.append(image)
            elif random.random() > 0.5:
                idx = random.randint(0, self.size - 1)
                old = self.images[idx].clone()
                self.images[idx] = image
                result.append(old)
            else:
                result.append(image)
        return torch.cat(result, dim=0)
