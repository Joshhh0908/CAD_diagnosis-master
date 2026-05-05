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
        length = self.input_shape[0]
        start, last = None, 0

        for i in range(labels_data.shape[0]):
            if start is not None:
                if labels_data[i] != last:
                    boxes.append([(start) / length, min((i+1) / length, 1.0)])
                    #note: end slice exclusive, ie lesion stops 1 slice before the end indicated
                    labels.append(last)
                    if labels_data[i] != 0:
                        start, last = i, labels_data[i]
                    else:
                        start, last = None, 0
            elif labels_data[i] != 0:
                    start, last = i, labels_data[i]

        if start is not None:
            boxes.append([(start) / length, 1.0])
            labels.append(last)

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes  = torch.tensor(boxes,  dtype=torch.float32)
        return {"labels": labels, "boxes": boxes}
    
    def sc_targets(self, labels_data, seq_length=32):
        segment_labels = torch.zeros(seq_length, dtype=torch.long)
        JOINT_TO_STEN   = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
        JOINT_TO_PLAQUE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}

        for i in range(0, labels_data.shape[0], 8):
            segment = labels_data[i:i+8]
            segment = set(np.unique(segment).tolist()) - {0}
            if not segment:
                segment_label = 0  # all background
            else:    
                segment_plaque = [JOINT_TO_PLAQUE[s] for s in segment]
                segment_sten   = [JOINT_TO_STEN[s]   for s in segment]
                segment_s = max(segment_sten)
                if 2 in segment_plaque or (3 in segment_plaque and 1 in segment_plaque):
                    segment_p = 2
                else:
                    segment_p = segment_plaque[0]

                segment_label = (segment_s - 1) * 3 + segment_p

            segment_idx = i // 8
            segment_labels[segment_idx] = segment_label
            # print(f"Segment {segment_idx}: {labels_data[i:i+8]} -> Label: {segment_label}")
        return {"labels": segment_labels}

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
            'od_target': self.detection_targets(labels),
            'sc_target': self.sc_targets(labels),
            'name': name
        }

    def __len__(self):
        return self.length


def collate_fn(batch):

    images, od_targets, sc_targets, names = [], [], [], []
    for item in batch:
        images.append(item['image'])
        od_targets.append(item['od_target'])
        sc_targets.append(item['sc_target'])
        names.append(item['name'])
    images = torch.stack(images, dim=0)

    return images, od_targets, sc_targets, names


