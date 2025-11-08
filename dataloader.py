import os
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class LOLDatasetTrain(Dataset):
    def __init__(self, root_dir, transform=None,patch_size=100):
        self.root_dir = root_dir
        self.low_dir = os.path.join(root_dir, "low")
        self.high_dir = os.path.join(root_dir, "high")
        self.low_images = sorted(os.listdir(self.low_dir))
        self.high_images = sorted(os.listdir(self.high_dir))
        self.transform = transform
        self.patch_size = patch_size

        assert len(self.low_images) == len(self.high_images)

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low = cv.imread(low_path)
        ref = cv.imread(high_path)
        low = cv.cvtColor(low, cv.COLOR_BGR2RGB)/255.
        ref = cv.cvtColor(ref, cv.COLOR_BGR2RGB)/255.

        h, w, _ = low.shape
        ps = self.patch_size
        if h >= ps and w >= ps:
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
            low[y:y + ps, x:x + ps, :] = ref[y:y + ps, x:x + ps, :]

        if self.transform:
            augmented = self.transform(image=low, image1=ref)
            low = augmented["image"]
            ref = augmented["image1"]

        low = low.float()
        ref = ref.float()
        return low, ref


class LOLDatasetTest(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.low_dir = os.path.join(root_dir, "low")
        self.high_dir = os.path.join(root_dir, "high")
        self.low_images = sorted(os.listdir(self.low_dir))
        self.high_images = sorted(os.listdir(self.high_dir))
        self.transform = transform

        assert len(self.low_images) == len(self.high_images)

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low = cv.imread(low_path)
        ref = cv.imread(high_path)
        low = cv.cvtColor(low, cv.COLOR_BGR2RGB) / 255.
        ref = cv.cvtColor(ref, cv.COLOR_BGR2RGB) / 255.

        if self.transform:
            augmented = self.transform(image=low, image1=ref)
            low = augmented["image"]
            ref = augmented["image1"]

        low = low.float()
        ref = ref.float()

        return low, ref




train_transform = A.Compose([A.Resize(height=512, width=512, interpolation=1),
                             A.HorizontalFlip(p=0.6),A.VerticalFlip(p=0.6),ToTensorV2()],
                            additional_targets={'image1': 'image'})
test_transform = A.Compose([A.Resize(height=512, width=512, interpolation=1),ToTensorV2()],
                            additional_targets={'image1': 'image'})




train_dataset = LOLDatasetTrain(root_dir="../Low_Light_Image_Enhancement/LOLdataset/train/", transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)



test_dataset = LOLDatasetTest(root_dir="../Low_Light_Image_Enhancement/LOLdataset/val/", transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)


