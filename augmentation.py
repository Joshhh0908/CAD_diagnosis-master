import glob
import os
import torch

from torch.utils import data
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk

import functions as funcs




class cubic_sequence_data(data.Dataset):
    def __init__(self, dataset_root=None, pattern='training', train_ratio=0.8,
                 input_shape=[480, 40, 40], window=[300, 900],
                 train_root=None, test_root=None):

        self.sitk = sitk
        self.input_shape = input_shape
        self.window = window

        if pattern == 'training' and train_root is not None:
            volume_root, label_root = train_root
        elif pattern != 'training' and test_root is not None:
            volume_root, label_root = test_root
        else:
            raise ValueError("Must provide train_root or test_root")

        self.volume_root = volume_root
        self.label_root = label_root

        self.volumes_file_list = sorted(glob.glob(os.path.join(volume_root, "*.nii.gz")))
        self.length = len(self.volumes_file_list)
        self.aug = True if pattern == 'training' else False


    def _fix_shape(self, vol, labels):
        Z, H, W = vol.shape
        target_Z, target_H, target_W = self.input_shape

        if H != target_H or W != target_W:
            vol = zoom(vol, (1, target_H / H, target_W / W), order=1)

        if Z > target_Z:
            vol    = zoom(vol, (target_Z / Z, 1, 1), order=1)
            labels = zoom(labels.astype(np.float32), target_Z / Z, order=0).astype(np.int32)
        elif Z < target_Z:
            pad = target_Z - Z
            vol    = np.pad(vol, ((0, pad), (0, 0), (0, 0)))
            labels = np.pad(labels, (0, pad))

        return vol, labels

    def detection_targets(self, labels_data):
        boxes, labels = [], []
        start, label, length, last = None, 0, self.input_shape[0], -1

        for i in range(labels_data.shape[0]):
            if start is not None:
                if labels_data[i] != last:
                    boxes.append([(start + 1) / length, min((i + 1) / length, 1.0)])
                    labels.append(label - 1)
                    if labels_data[i] != 0:
                        start, label, last = i, labels_data[i], labels_data[i]
                    else:
                        start, label, last = None, 0, -1
                else:
                    continue
            else:
                if labels_data[i] == 0:
                    start, label, last = None, 0, -1
                else:
                    start, label, last = i, labels_data[i], labels_data[i]

        if start is not None:
            boxes.append([(start + 1) / length, 1.0])
            labels.append(label - 1)

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes  = torch.tensor(boxes,  dtype=torch.float32)
        return {"labels": labels, "boxes": boxes}

    def __getitem__(self, index):
        vf   = self.volumes_file_list[index]
        name = os.path.basename(vf).replace(".nii.gz", "")
        lf   = os.path.join(self.label_root, name + ".txt")  # single combined label file

        vol = self.sitk.GetArrayFromImage(self.sitk.ReadImage(vf)).astype(np.float32)
        labels = np.loadtxt(lf, dtype=np.int32)

        # normalize
        hu_min = self.window[0]
        hu_max = self.window[1]
        vol = funcs.normalize_ct_data(vol, hu_min=hu_min, hu_max=hu_max)

        # resize
        vol, labels = self._fix_shape(vol, labels)

        return {
            'image':  torch.tensor(vol, dtype=torch.float32),
            'target': self.detection_targets(labels)
        }

    def __len__(self):
        return self.length


def collate_fn(batch):

    images, targets = [], []
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    images = torch.stack(images, dim=0)

    return images, targets


