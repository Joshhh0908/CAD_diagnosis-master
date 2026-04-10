import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import zoom, rotate
from config import cfg


class CPRDataset(Dataset):
    def __init__(self, volume_dir, augment=False):
        #
        self.augment = augment
        self.paths = []
        for vf in sorted(glob.glob(os.path.join(volume_dir, "*.nii.gz"))):
            name = os.path.basename(vf).replace(".nii.gz", "")
            lf = os.path.join(volume_dir.replace("volumes","lumen_masks"), name + ".nii.gz")
            vef = os.path.join(volume_dir.replace("volumes", "vessel_masks"), name + ".nii.gz")
            sf = os.path.join(volume_dir.replace("volumes","labels"), name + "_stenosis.txt")
            pf = os.path.join(volume_dir.replace("volumes","labels"), name + "_plaque.txt")
            if all(os.path.exists(f) for f in [lf, vef, sf, pf]):
                self.paths.append((vf, lf, vef, sf, pf))

    def __len__(self):
        return len(self.paths)

    def _fix_shape(self, vol, ls, lp):
        Z, H, W = vol.shape

        # H, W: already handled by center crop in __getitem__
        # kept for safety (e.g. direct _fix_shape calls in test scripts)
        if H != cfg.H or W != cfg.W:
            vol = zoom(vol, (1, cfg.H / H, cfg.W / W), order=1)

        # Z: resize if longer, pad if shorter
        if Z > cfg.Z:
            vol = zoom(vol, (cfg.Z / Z, 1, 1), order=1)
            ls  = zoom(ls.astype(np.float32), cfg.Z / Z, order=0).astype(np.int64)
            lp  = zoom(lp.astype(np.float32), cfg.Z / Z, order=0).astype(np.int64)
        elif Z < cfg.Z:
            pad = cfg.Z - Z
            vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)))  # pad zeros
            ls  = np.pad(ls,  (0, pad))
            lp  = np.pad(lp,  (0, pad))

        valid = np.zeros(cfg.Z, dtype=np.float32)
        valid[:min(Z, cfg.Z)] = 1.0

        # ls = np.clip(ls, 0, cfg.num_stenosis - 1)
        # lp = np.clip(lp, 0, cfg.num_plaque   - 1)
        return vol, ls, lp, valid

    def _fix_mask(self, mask):
        """Resize/pad mask along Z to match cfg.Z (nearest-neighbour to preserve labels)."""
        Z = mask.shape[0]
        if Z > cfg.Z:
            mask = zoom(mask, (cfg.Z / Z, 1, 1), order=0)
        elif Z < cfg.Z:
            pad = cfg.Z - Z
            mask = np.pad(mask, ((0, pad), (0, 0), (0, 0)))
        return mask

    @staticmethod
    def _center_crop(vol, crop_h, crop_w):
        """
        Crop (Z, H, W) → (Z, crop_h, crop_w) centered at (H//2, W//2).
        CPR volumes are centered on the vessel centerline, so center crop
        keeps the lumen and removes peripheral tissue.
        If volume is smaller than crop size, pad symmetrically.
        """
        Z, H, W = vol.shape
        # Symmetric padding: distribute extra pixels evenly on both sides
        if H < crop_h:
            pad_h = crop_h - H
            vol = np.pad(vol, ((0, 0), (pad_h // 2, (pad_h + 1) // 2), (0, 0)))
        if W < crop_w:
            pad_w = crop_w - W
            vol = np.pad(vol, ((0, 0), (0, 0), (pad_w // 2, (pad_w + 1) // 2)))
        Z, H, W = vol.shape
        ch = H // 2
        cw = W // 2
        h0, h1 = ch - crop_h // 2, ch - crop_h // 2 + crop_h
        w0, w1 = cw - crop_w // 2, cw - crop_w // 2 + crop_w
        return vol[:, h0:h1, w0:w1]

    def __getitem__(self, idx):
        vf, lf, vef, sf, pf = self.paths[idx]

        vol  = sitk.GetArrayFromImage(sitk.ReadImage(vf)).astype(np.float32)
        lumen_mask = sitk.GetArrayFromImage(sitk.ReadImage(lf)).astype(np.float32)
        vessel_mask = sitk.GetArrayFromImage(sitk.ReadImage(vef)).astype(np.float32)
        vol  = np.clip(vol, -300, 900)
        vol  = (vol - vol.mean()) / (vol.std() + 1e-6)

        # center crop to cfg.H × cfg.W BEFORE _fix_shape
        # CPR centerline is at (H//2, W//2) → preserves lumen, removes periphery
        vol  = self._center_crop(vol,  cfg.H, cfg.W)
        lumen_mask = self._center_crop(lumen_mask, cfg.H, cfg.W)
        vessel_mask = self._center_crop(vessel_mask, cfg.H, cfg.W)

        ls = np.loadtxt(sf, dtype=np.int64)
        lp = np.loadtxt(pf, dtype=np.int64)

        orig_Z = vol.shape[0]
        vol, ls, lp, valid = self._fix_shape(vol, ls, lp)
        lumen_mask = self._fix_mask(lumen_mask)
        vessel_mask = self._fix_mask(vessel_mask)
        # FIX 1: define lesion_exist from labels before the augment block
        lesion_exist = ((ls > 0) | (lp > 0)).astype(np.float32)  # (Z,)

        if self.augment:
            # ── 1. Z flip ─────────────────────────
            if np.random.rand() > 0.5:
                vol          = vol[::-1].copy()
                ls           = ls[::-1].copy()
                lp           = lp[::-1].copy()
                valid        = valid[::-1].copy()
                lumen_mask         = lumen_mask[::-1].copy()
                vessel_mask = vessel_mask[::-1].copy()
                lesion_exist = lesion_exist[::-1].copy()

            # ── 2. HU noise ────────────────
            # FIX 2: cast the whole expression, not the scalar
            if np.random.rand() > 0.5:
                vol = (vol + np.random.normal(0, 0.005)).astype(np.float32)

            # ── 3. Gamma ───────────────
            if np.random.rand() > 0.5:
                gamma = np.random.uniform(0.95, 1.05)
                vol = (np.sign(vol) * (np.abs(vol) ** gamma)).astype(np.float32)

            # ── 4. Random in-plane rotation (H×W axes) ───────────────
            # Rotates each Z-slice around the vessel center; uses nearest-neighbour
            # for mask and bilinear for the volume.
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)  # degrees
                # vol/mask: (Z, H, W) — rotate in axes (1,2), i.e. the H×W plane
                vol  = rotate(vol,  angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
                lumen_mask  = rotate(lumen_mask,  angle, axes=(1, 2), reshape=False, order=0, mode='nearest')
                vessel_mask = rotate(vessel_mask, angle, axes=(1, 2), reshape=False, order=0, mode='nearest')
                # FIX 2: binarize after rotation — float arithmetic with order=0
                # can leave values like 0.9999 or 0.0001 instead of exact 0/1
                lumen_mask  = (lumen_mask  > 0.5).astype(np.float32)
                vessel_mask = (vessel_mask > 0.5).astype(np.float32)
                # ls, lp, valid, lesion_exist are 1-D along Z — unaffected by spatial rotation

            # # ── 5. Random Z-shift (simulate different acquisition start points) ───────────────
            # if np.random.rand() > 0.5:
            #     shift = np.random.randint(-3, 3)
            #     vol          = np.roll(vol,  shift, axis=0)
            #     ls           = np.roll(ls,   shift, axis=0)
            #     lp           = np.roll(lp,   shift, axis=0)
            #     valid        = np.roll(valid, shift, axis=0)
            #     lumen_mask   = np.roll(lumen_mask,  shift, axis=0)
            #     vessel_mask  = np.roll(vessel_mask, shift, axis=0)
            #     lesion_exist = np.roll(lesion_exist, shift, axis=0)
            #     # FIX 1: zero out ALL arrays in the shifted region, not just valid/lesion_exist
            #     # np.roll wraps data from the opposite end — those voxels are phantom data
            #     if shift > 0:
            #         vol[:shift]          = 0
            #         ls[:shift]           = 0
            #         lp[:shift]           = 0
            #         valid[:shift]        = 0
            #         lumen_mask[:shift]   = 0
            #         vessel_mask[:shift]  = 0
            #         lesion_exist[:shift] = 0
            #     elif shift < 0:
            #         vol[shift:]          = 0
            #         ls[shift:]           = 0
            #         lp[shift:]           = 0
            #         valid[shift:]        = 0
            #         lumen_mask[shift:]   = 0
            #         vessel_mask[shift:]  = 0
            #         lesion_exist[shift:] = 0

        # Derive lesion_start / lesion_end from lesion_exist AFTER all augmentations
        # (Z-flip and Z-shift may have changed the position of the lesion)
        # -1 indicates no lesion present in this sample
        nonzero = np.where(lesion_exist > 0)[0]
        if len(nonzero) > 0:
            lesion_start = int(nonzero[0])
            lesion_end   = int(nonzero[-1])
        else:
            lesion_start = -1
            lesion_end   = -1

        return {
            'volume':        torch.FloatTensor(vol).unsqueeze(1),            # (Z, 1, H, W)
            'lumen_mask':    torch.FloatTensor(lumen_mask).unsqueeze(1),     # (Z, 1, H, W)
            'vessel_mask':   torch.FloatTensor(vessel_mask).unsqueeze(1),    # (Z, 1, H, W)
            'labels_s':      torch.LongTensor(ls),                           # (Z,)
            'labels_p':      torch.LongTensor(lp),                           # (Z,)
            'valid_mask':    torch.FloatTensor(valid),                       # (Z,)
            'lesion_exist':  torch.FloatTensor(lesion_exist),                # (Z,)
            'lesion_start':  torch.tensor(lesion_start, dtype=torch.long),  # scalar, -1=no lesion
            'lesion_end':    torch.tensor(lesion_end,   dtype=torch.long),  # scalar, -1=no lesion
            'name':          os.path.basename(vf).replace('.nii.gz', ''),
        }