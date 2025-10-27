import albumentations as A
import cv2 as cv
import torch
import numpy as np

def get_train_augs(IMG_SIZE):
  return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv.INTER_NEAREST),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=45,p=0.5),
                    ])

def get_test_augs(IMG_SIZE):
  return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv.INTER_NEAREST),])

class OCTSegmentation(torch.utils.data.Dataset):

  def __init__(self,all_images,all_masks,augmentations):
    self.data = list(zip(all_images,all_masks))
    self.augmentations = augmentations

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    image, mask = self.data[index]
    image = image.astype(float) / 255.0
    mask = mask.astype(float)

    if self.augmentations:
      data = self.augmentations(image=image,mask=mask)
      image,mask= data['image'],data['mask']

    return torch.from_numpy(image).unsqueeze(0).float(),torch.from_numpy(mask).unsqueeze(0).float()