from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import numpy as np
from utils import *
import cv2
import os
import albumentations as albu
from albumentations.pytorch import ToTensor


class OpticalFlowDataset(Dataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None,co_aug_transforms=None):
        self.root = root
        self.frames_transforms = frames_transforms
        self.frames_aug_transforms = frames_aug_transforms
        self.co_aug_transforms = co_aug_transforms
        self.target3 = {'image0': 'image', 'image1': 'image', 'image2': 'image'}
        self.target2 = {'image0': 'image', 'image1': 'image'}
        self.imgPairs = []
        self.flows = []
        self.func = readflo

    def __getitem__(self, idx):

        flow = self.func(self.flows[idx])
        frame1 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][0]), cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][1]), cv2.COLOR_BGR2RGB)

        if self.co_aug_transforms is not None:
            transformed = albu.Compose(self.co_aug_transforms, p=1, additional_targets=self.target3)(image=frame1,
                                                                                                     image0=frame2,
                                                                                                     image1=flow)
            frame1 = transformed['image']
            frame2 = transformed['image0']
            flow = transformed['image1']

        if self.frames_aug_transforms is not None:
            transformed = albu.Compose(self.frames_aug_transforms, p=1, additional_targets=self.target2)(image=frame1,
                                                                                                         image0=frame2)
            frame1 = transformed['image']
            frame2 = transformed['image0']

        if self.frames_transforms is not None:

            frame1 = self.frames_transforms(image=frame1)['image']
            frame2 = self.frames_transforms(image=frame2)['image']

        flow = torch.from_numpy(flow.transpose(2, 0, 1).copy())
        frames = torch.cat((frame1, frame2), dim=0)
        return frames, flow

    def __len__(self):
        return len(self.imgPairs)


class SintelDataset(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        scenes = sorted(os.listdir(os.path.join(self.root, "final")))
        genres = ['albedo', 'clean', 'final']
        for scene in scenes:
            frame_names = [path.split('.')[0] for path in sorted(os.listdir(os.path.join(self.root, "final", scene)))]
            for i in range(len(frame_names) - 1):
                self.flows += [os.path.join(self.root, 'flow', scene, frame_names[i] + ".flo")] * len(genres)

                self.imgPairs += [[os.path.join(self.root, genre, scene, frame_names[i] + '.png'),
                                   os.path.join(self.root, genre, scene, frame_names[i + 1] + '.png')] for genre in
                                  genres]

        assert (len(self.imgPairs) == len(self.flows))


class FlyingChairs(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        for file_name in sorted(os.listdir(self.root)):
            if "flow" in file_name:
                self.flows.append(os.path.join(self.root, file_name))
            elif "img1" in file_name:
                self.imgPairs.append([os.path.join(self.root, file_name)])
            else:
                self.imgPairs[-1].append(os.path.join(self.root, file_name))

        assert (len(self.imgPairs) == len(self.flows))


class Chairs3D(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        self.func = readPFM
        for flow, im1, im2 in zip(sorted(os.listdir(os.path.join(self.root, "flow"))),
                                  sorted(os.listdir(os.path.join(self.root, "t0"))),
                                  sorted(os.listdir(os.path.join(self.root, "t1")))):
            self.flows.append(os.path.join(self.root, "flow", flow))
            self.imgPairs.append([os.path.join(self.root,"t0", im1)])
            self.imgPairs[-1].append(os.path.join(self.root,"t1", im2))

        assert (len(self.imgPairs) == len(self.flows))


class OceanData(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        file_names = sorted(os.listdir(self.root))

        for i in range(0, len(file_names) - 1):
            self.imgPairs.append([os.path.join(self.root, file_names[i]), os.path.join(self.root, file_names[i + 1])])

    def __getitem__(self, idx):

        frame1 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][0]), cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][1]), cv2.COLOR_BGR2RGB)

        if self.co_aug_transforms is not None:
            transformed = albu.Compose(self.co_aug_transforms, p=1, additional_targets=self.target2)(image=frame1,
                                                                                                     image0=frame2)
            frame1 = transformed['image']
            frame2 = transformed['image0']

        if self.frames_aug_transforms is not None:
            transformed = albu.Compose(self.frames_aug_transforms, p=1, additional_targets=self.target2)(image=frame1,
                                                                                                         image0=frame2)
            frame1 = transformed['image']
            frame2 = transformed['image0']

        if self.frames_transforms is not None:

            frame1 = self.frames_transforms(image=frame1)['image']
            frame2 = self.frames_transforms(image=frame2)['image']

        frames = torch.cat((frame1, frame2), dim=0)
        return frames,


def getDataloaders(batch_size, root='../sintel/training', frames_transforms=None, frames_aug_transforms=None,
                   co_aug_transforms=None):

    test_size = 50
    if "sintel" in root:
        train_dataset = SintelDataset(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = SintelDataset(root, frames_transforms)
        val_size = 133
    elif "ocean" in root:
        train_dataset = OceanData(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = OceanData(root, frames_transforms)
        val_size = 100
    elif "Flying" in root:
        print("Using Flying Chairs dataset")
        train_dataset = FlyingChairs(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = FlyingChairs(root, frames_transforms)
        val_size = 640
        test_size = 200
    else:
        print("Using Chairs3D dataset")
        train_dataset = Chairs3D(os.path.join(root, "train"), frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = Chairs3D(os.path.join(root, "train"), frames_transforms)
        test_dataset = Chairs3D(os.path.join(root, "test"), frames_transforms)
        val_size = 640
        test_size = 0

    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_idx, valid_idx, test_idx = indices[val_size + test_size:], indices[:val_size], indices[
                                                                                         val_size:test_size + val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    if len(test_idx) > 0:
        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=test_sampler,
                                 pin_memory=torch.cuda.is_available(), num_workers=4)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=torch.cuda.is_available(),
                                 num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              pin_memory=torch.cuda.is_available(), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            pin_memory=torch.cuda.is_available(), num_workers=4)

    return train_loader, val_loader, test_loader
