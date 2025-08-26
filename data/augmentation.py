from torchvision import transforms
from PIL import Image


class DataAugmentationDINO(object):
    """
    Defines the complex data augmentation pipeline for DINO, including global and local crops.
    """
    def __init__(self, global_crops_scale, local_crops_scale, n_local_crops):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # First global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            normalize,
        ])
        # Second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomSolarize(128, p=0.2),
            normalize,
        ])
        # Local crops
        self.n_local_crops = n_local_crops
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.n_local_crops):
            crops.append(self.local_transfo(image))
        return crops
    
    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), werecommand using a wider range of scale")
        parser.add_argument('--n_local_crops', type=int, default=8, help="Number of small local views to generate. Set this parameter to 0 to disable multi-crop training.")
        parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="Scale range of the cropped image before resizing, relatively to the origin image. Used for small local view cropping of multi-crop.""")
        return parser