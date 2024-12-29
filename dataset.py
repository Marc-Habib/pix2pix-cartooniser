import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CartoonDataset(Dataset):
    """
    A custom Dataset class that:
      1. Scans 'data/<mode>/real/' and 'data/<mode>/cartoon/'.
      2. Matches images by filename, e.g. image001.jpg in both real/ and cartoon/.
      3. Transforms them to a consistent size (e.g., 256x256).
      4. Normalises them to a range [-1, +1].
    """
    def __init__(self, root_dir, mode="train", transform=None, image_size=256):
        super().__init__()
        self.root_dir = root_dir      # e.g., "data"
        self.mode = mode              # "train" or "test"
        self.real_dir = os.path.join(root_dir, mode, "real")
        self.cartoon_dir = os.path.join(root_dir, mode, "cartoon")

        # List all filenames in real/ and cartoon/ and sort so they match by index
        self.real_images = sorted(os.listdir(self.real_dir))
        self.cartoon_images = sorted(os.listdir(self.cartoon_dir))

        self.transform = transform
        if transform is None:
            # Default transforms: 
            #  1) Resize (image_size x image_size)
            #  2) Convert to Tensor
            #  3) Normalise to [-1,1] (mean=0.5, std=0.5 for each channel)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

    def __len__(self):
        # The number of examples is based on how many real images we have
        return len(self.real_images)

    def __getitem__(self, idx):
        real_name = self.real_images[idx]
        cartoon_name = self.cartoon_images[idx]

        real_path = os.path.join(self.real_dir, real_name)
        cartoon_path = os.path.join(self.cartoon_dir, cartoon_name)

        # Load images with PIL
        real_img = Image.open(real_path).convert("RGB")
        cartoon_img = Image.open(cartoon_path).convert("RGB")

        # Apply the same transform to both
        real_img = self.transform(real_img)
        cartoon_img = self.transform(cartoon_img)

        # Return a tuple: (real_image_tensor, cartoon_image_tensor)
        return real_img, cartoon_img
