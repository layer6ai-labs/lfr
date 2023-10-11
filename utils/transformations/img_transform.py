import torchvision.transforms as transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def KvasirTransform(augmentation=False, contrastive=False):
    if not augmentation:
        return None
    augmentations = [
                    transforms.RandomResizedCrop((80,100)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size = (9, 9))
                    ]
    img_transforms = transforms.Compose(augmentations)
    if contrastive:
        img_transforms = TwoCropsTransform(img_transforms)
    return img_transforms
