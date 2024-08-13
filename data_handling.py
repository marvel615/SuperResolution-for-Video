import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import utils


def load_image(file_path):
    # Open the image file
    with Image.open(file_path) as img:
        # Convert the image to RGB if it's not in that mode
        img = img.convert('RGB')
        return img


##########################################################
# Datasets
##########################################################

class ZSSRDataset(Dataset):
    def __init__(self, img_path, scale_factor=2, transform=transforms.ToTensor()):
        self.img_path = img_path
        self.transform = transform

        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = load_image(self.img_path[idx])
        img = self.transform(img)

        img_sr = img.clone()
        img_lr = utils.rr_resize(img, scale_factors=1.0/self.scale_factor)

        return {"SR": img_sr, "LR": img_lr}


class VideoFramesDataset(Dataset):
    def __init__(self, low_res_frames, high_res_frames, transform=None):
        self.low_res_frames = low_res_frames
        self.high_res_frames = high_res_frames
        self.transform = transform

    def __len__(self):
        return len(self.low_res_frames)

    def __getitem__(self, idx):
        low_res = self.low_res_frames[idx] # list of 3 PIL frames LR
        high_res = self.high_res_frames[idx] # list of 3 PIL frames SR
        if self.transform:
            low_res = [self.transform(low) for low in low_res] # list of tensor
            high_res = [self.transform(high) for high in high_res]


        return torch.stack(low_res), torch.stack(high_res)

##########################################################
# Transforms 
##########################################################

def advanced_trans():
  """transforms used in the advanced case for training.

  Returns:
    output (callable) - A transformation that recieves a PIL image, converts it
    to torch.Tensor, and takes the FourCrops of this
    random crop. The result's shape is 4 x C x H x W.

  Note: you may explore different augmentations for your original implementation.
  """

  def transform(image):
    torch_img = transforms.ToTensor()(image)
    ec = FourCrops()(torch_img)

    return ec


  return transform

class FourCrops:
  """Generate all the possible crops using combinations of
  [0 and 180 degrees rotations,  horizontal flips and vertical flips].
  In total there are 4 options."""

  def __init__(self):
    pass

  def __call__(self, sample):
    """
    Args:
      sample (torch.Tensor) - image to be transformed.
      Has shape `(num_channels, height, width)`.
    Returns:
      output (List(torch.Tensor)) - A list of 8 tensors containing the different
      flips and rotations of the original image. Each tensor has the same size as
      the original image, possibly transposed in the spatial dimensions.
    """
    # BEGIN SOLUTION
    output = []

    # Original image
    output.append(sample)

    # Horizontal flip
    flipped_horizontal = torch.flip(sample, dims=(2,))
    output.append(flipped_horizontal)

    # Vertical flip
    flipped_vertical = torch.flip(sample, dims=(1,))
    output.append(flipped_vertical)

    # 180-degree rotation
    rotated_180 = torch.rot90(sample, k=2, dims=(1, 2))
    output.append(rotated_180)

    return torch.stack(output)




