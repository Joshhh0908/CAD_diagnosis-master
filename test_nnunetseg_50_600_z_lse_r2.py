"""
Test script for NNUNet3D lesion segmentation.

Definitions
-----------
Vessel-level  (one result per volume):
    GT   positive  = volume has at least 1 lesion slice
    Pred positive  = model predicts at least 1 slice as lesion in that volume
    → standard 2-class confusion matrix over all volumes

Lesion-level  (one result per labelled stenosis segment):
    A "lesion" = one contiguous run of the SAME non-zero stenosis label in ls.
    Segments with different labels that happen to be adjacent are counted as
    SEPARATE lesions, consistent with the CSV annotation statistics.
    Each lesion instance is either:
        detected (TP)  – model predicts >=1 slice inside that segment as positive
        missed   (FN)  – no slice inside the segment is predicted positive
    False alarms (FP) = contiguous predicted-positive runs that do not overlap
                        with any GT lesion segment in that volume.
    TN = not meaningful at instance level; reported at vessel level instead.

Visualization
-------------
For each volume a PNG is saved showing the vessel in longitudinal view
(centre column of the CPR volume unrolled along Z).
  Red   overlay  = predicted lesion slices
  Green outline  = GT lesion slices
  Cyan  outline  = GT lesion slices that were MISSED by the model
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk
from dataset.dataset_50_600 import CPRDataset
from method.model_nnunetseg_ms_z_lse import NNUNet3D
from config_50_600 import cfg

# Utility

def get_labeled_segments(label_array):
    """
    Return list of (start, end, label) for each contiguous run of the SAME
    non-zero label value.

    Adjacent segments with DIFFERENT labels are treated as separate lesions,
    matching the original annotation format where each (start, end, label)
    triplet in the CSV is one independent lesion entry.

    Example:
        [0, 0, 1, 1, 2, 2, 0]  ->  [(2, 3, 1), (4, 5, 2)]  # two lesions
        [0, 0, 1, 1, 1, 1, 0]  ->  [(2, 5, 1)]              # one lesion
    """
    segments = []
    in_seg = False
    current_label = 0
    start = 0
    for i, v in enumerate(label_array):
        v = int(v)
        if v != 0 and not in_seg:
            start, in_seg, current_label = i, True, v
        elif v != 0 and in_seg and v != current_label:
            # label changed -> close current, open new
            segments.append((start, i - 1, current_label))
            start, current_label = i, v
        elif v == 0 and in_seg:
            segments.append((start, i - 1, current_label))
            in_seg = False
    if in_seg:
        segments.append((start, len(label_array) - 1, current_label))
    return segments


def get_contiguous_segments(binary_array):
    """Return list of (start, end) inclusive runs of 1s in a 1-D binary array.
    Used for predicted segments which carry no label grade."""
    segments, in_seg = [], False
    for i, v in enumerate(binary_array):
        if v == 1 and not in_seg:
            start, in_seg = i, True
        elif v == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary_array) - 1))
    return segments


def segments_overlap(seg, pred_array):
    """Return True if any slice in seg=(start, end, ...) is predicted positive."""
    return pred_array[seg[0]:seg[1] + 1].max() == 1


def print_confusion_matrix(cm, title="Confusion Matrix"):
    print(f"\n  {title}")
    print(f"                   Pred NEG    Pred POS")
    print(f"  Actual NEG   :  {cm[0,0]:>9d}   {cm[0,1]:>9d}")
    print(f"  Actual POS   :  {cm[1,0]:>9d}   {cm[1,1]:>9d}")


def compute_metrics(cm):
    TN, FP = int(cm[0, 0]), int(cm[0, 1])
    FN, TP = int(cm[1, 0]), int(cm[1, 1])
    sensitivity = TP / (TP + FN + 1e-9)
    specificity = TN / (TN + FP + 1e-9)
    precision   = TP / (TP + FP + 1e-9)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)
    accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                sensitivity=sensitivity, specificity=specificity,
                precision=precision, f1=f1, accuracy=accuracy)


def print_metrics(m, title="Metrics"):
    print(f"\n  {title}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  {'Sensitivity (Recall)':<25}: {m['sensitivity']:.4f}")
    print(f"  {'Specificity':<25}: {m['specificity']:.4f}")
    print(f"  {'Precision':<25}: {m['precision']:.4f}")
    print(f"  {'F1 Score':<25}: {m['f1']:.4f}")
    print(f"  {'Accuracy':<25}: {m['accuracy']:.4f}")


# Visualization

def save_visualization(vol_np, slice_gt, slice_pred, gt_segs, valid_len, name, save_dir):
    """
    Save a 3-row visualization:
      Row 1 – CPR longitudinal image (greyscale)
      Row 2 – GT  bar: black=normal, red=abnormal
      Row 3 – Pred bar: black=normal, red=abnormal

    vol_np    : (Z, H, W)  normalised CT volume
    slice_gt  : (valid_len,) int32  0/1  binary GT mask (for display)
    slice_pred: (valid_len,) int32  0/1
    gt_segs   : list of (start, end, label) from get_labeled_segments
    valid_len : real slice count
    """
    Z, H, W = vol_np.shape
    cw = W // 2

    longit = vol_np[:valid_len, :, cw].T
    vmin, vmax = np.percentile(longit, 1), np.percentile(longit, 99)
    longit_norm = np.clip((longit - vmin) / (vmax - vmin + 1e-6), 0, 1)

    pred_segs = get_contiguous_segments(slice_pred)

    v_gt   = int(slice_gt.max())  if valid_len > 0 else 0
    v_pred = int(slice_pred.max()) if valid_len > 0 else 0
    v_ok   = "✓" if v_gt == v_pred else "✗"

    n_lesions  = len(gt_segs)
    n_detected = sum(1 for s in gt_segs if segments_overlap(s, slice_pred))
    n_missed   = n_lesions - n_detected
    n_fp       = sum(1 for ps in pred_segs
                     if not any(not (ps[1] < gs[0] or ps[0] > gs[1])
                                for gs in gt_segs))

    title = (f"{name}  |  vessel gt={v_gt} pred={v_pred} {v_ok}  |  "
             f"lesions={n_lesions} detected={n_detected} "
             f"missed={n_missed} fp={n_fp}")

    def make_bar_img(mask, bar_h=30):
        img = np.zeros((bar_h, valid_len, 3), dtype=np.float32)
        for z in range(valid_len):
            if mask[z] == 1:
                img[:, z, 0] = 1.0
        return img

    bar_gt   = make_bar_img(slice_gt)
    bar_pred = make_bar_img(slice_pred)

    fig_w  = max(14, valid_len // 8)
    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_w, 7),
        gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.35}
    )
    fig.suptitle(title, fontsize=9, y=1.01)

    imshow_kw = dict(aspect="auto", origin="upper", interpolation="nearest")

    axes[0].imshow(longit_norm, cmap="gray", **imshow_kw)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(bar_gt, **imshow_kw)
    axes[1].set_ylabel("GT", fontsize=9, labelpad=4, va="center")
    axes[1].set_yticks([])
    axes[1].set_xticks(np.arange(0, valid_len, max(1, valid_len // 6)).tolist())
    axes[1].tick_params(axis="x", labelsize=8)
    legend_els = [mpatches.Patch(facecolor="black", label="normal"),
                  mpatches.Patch(facecolor="red",   label="abnormal")]
    axes[1].legend(handles=legend_els, loc="upper right", fontsize=7, framealpha=0.8)

    axes[2].imshow(bar_pred, **imshow_kw)
    axes[2].set_ylabel("Pred", fontsize=9, labelpad=4, va="center")
    axes[2].set_yticks([])
    axes[2].set_xticks(np.arange(0, valid_len, max(1, valid_len // 6)).tolist())
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].set_xlabel("Z (slice index)", fontsize=9)
    axes[2].legend(handles=legend_els, loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# Test loop

def test(model, test_dir, device, vis_dir):
    model.eval()

    vessel_gt_list, vessel_pred_list = [], []
    lesion_TP, lesion_FN, lesion_FP, lesion_TN = 0, 0, 0, 0
    per_sample = []
    test_ids = sorted(os.listdir(test_dir))

    print(f"Test samples : {len(test_ids)}")
    print(f"Visualizations → {vis_dir}")

    with torch.no_grad():
        for each_id in tqdm(test_ids):
            vol_path = os.path.join(test_dir, each_id)
            name = each_id.replace('.nii.gz', '')
            label_dir = test_dir.replace('volumes', 'labels')
            sf = os.path.join(label_dir, name + '_stenosis.txt')
            pf = os.path.join(label_dir, name + '_plaque.txt')

            # load & preprocess volume
            vol = sitk.GetArrayFromImage(sitk.ReadImage(vol_path)).astype(np.float32)
            vol = np.clip(vol, -300, 900)
            vol = (vol - vol.mean()) / (vol.std() + 1e-6)
            vol = CPRDataset._center_crop(vol, cfg.H, cfg.W)   # (Z, H, W)

            ls = np.loadtxt(sf, dtype=np.int64)
            lp = np.loadtxt(pf, dtype=np.int64)

            _ds = CPRDataset.__new__(CPRDataset)
            vol, ls, lp, valid_np = _ds._fix_shape(vol, ls, lp)

            valid_len = int(valid_np.sum())

            # Binary GT for vessel-level detection and visualisation
            lesion_exist = ((ls > 0) | (lp > 0)).astype(np.int32)
            slice_gt     = lesion_exist[:valid_len]

            # GT lesion segments: extract from the full ls array so no segment
            # is silently dropped by valid_len truncation.
            # Segments that start before valid_len are kept; their end is clipped
            # to valid_len-1 if needed. This keeps the count consistent with the
            # per-triplet CSV annotation statistics.
            gt_segs = [
                (s, min(e, valid_len - 1), lbl)
                for s, e, lbl in get_labeled_segments(ls)
                if s < valid_len
            ]

            # Model inference
            inp = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0).to(device)
            logits     = model(inp)                  # (1, 2, Z, H, W)
            pred_voxel = logits.argmax(dim=1).cpu()  # (1, Z, H, W)

            slice_pred = (pred_voxel[0, :valid_len]
                          .amax(dim=(-2, -1))
                          .numpy().astype(np.int32))

            pred_segs = get_contiguous_segments(slice_pred)

            # Vessel-level
            v_gt   = int(slice_gt.max())
            v_pred = int(slice_pred.max())
            vessel_gt_list.append(v_gt)
            vessel_pred_list.append(v_pred)

            # Lesion-instance-level
            vol_TP, vol_FN, vol_FP = 0, 0, 0

            if v_gt == 0 and v_pred == 0:
                lesion_TN += 1

            for seg in gt_segs:
                if segments_overlap(seg, slice_pred):
                    lesion_TP += 1; vol_TP += 1
                else:
                    lesion_FN += 1; vol_FN += 1

            for pseg in pred_segs:
                overlaps_any = any(
                    not (pseg[1] < gseg[0] or pseg[0] > gseg[1])
                    for gseg in gt_segs
                )
                if not overlaps_any:
                    lesion_FP += 1; vol_FP += 1

            per_sample.append({
                'name':           name,
                'valid_slices':   valid_len,
                'n_gt_lesions':   len(gt_segs),
                'n_pred_lesions': len(pred_segs),
                'lesion_TP': vol_TP, 'lesion_FN': vol_FN, 'lesion_FP': vol_FP,
                'vessel_gt': v_gt,   'vessel_pred': v_pred,
            })

            save_visualization(
                vol_np     = vol,
                slice_gt   = slice_gt,
                slice_pred = slice_pred,
                gt_segs    = gt_segs,
                valid_len  = valid_len,
                name       = name,
                save_dir   = vis_dir,
            )

    return per_sample, vessel_gt_list, vessel_pred_list, lesion_TP, lesion_FN, lesion_FP, lesion_TN


# Entry point

def main():
    device     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    test_dir   = "/mnt/nas4/diskm/wangxh/ctca_no/dataset/test_geo_02mm_clean/volumes/"
    ckpt_path  = "/mnt/nas4/diskm/wangxh/ctca_no/result/chk_nnunetseg_600_50_z_lse_r2_b1/best_model.pth"
    output_dir = "/mnt/nas4/diskm/wangxh/ctca_no/result/chk_nnunetseg_600_50_z_lse_r2_b1/"
    vis_dir    = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir,    exist_ok=True)

    model = NNUNet3D(in_ch=1, num_classes=2).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Checkpoint   : epoch={ckpt['epoch']}, val_dice={ckpt['val_dice']:.4f}")

    per_sample, vessel_gt, vessel_pred, l_TP, l_FN, l_FP, l_TN = \
        test(model, test_dir, device, vis_dir)

    # Vessel-level
    cm_v = confusion_matrix(vessel_gt, vessel_pred, labels=[0, 1])
    m_v  = compute_metrics(cm_v)

    print("\nVESSEL-LEVEL  (one result per volume)")
    print(f"  Total vessels : {len(vessel_gt)}")
    print(f"  GT positive   : {sum(vessel_gt)}")
    print_confusion_matrix(cm_v, "Vessel-Level Confusion Matrix")
    print_metrics(m_v, "Vessel-Level Metrics")

    # Lesion-instance-level
    cm_l = np.array([[l_TN, l_FP], [l_FN, l_TP]])
    m_l  = compute_metrics(cm_l)

    print("\nLESION-LEVEL  (one result per labelled stenosis segment)")
    print(f"  Total GT lesions   : {l_TP + l_FN}")
    print(f"  Total pred lesions : {l_TP + l_FP}")
    print_confusion_matrix(cm_l, "Lesion-Level Confusion Matrix")
    print_metrics(m_l, "Lesion-Level Metrics")

    # Per-sample table
    print("\nPER-SAMPLE RESULTS")
    print(f"  {'Name':<35} {'ValidZ':>6} {'GT_L':>5} {'Pred_L':>6} "
          f"{'TP':>4} {'FN':>4} {'FP':>4} {'V_GT':>5} {'V_Pred':>6} {'V_OK':>6}")
    for r in per_sample:
        v_ok = "YES" if r['vessel_gt'] == r['vessel_pred'] else "NO"
        print(f"  {r['name']:<35} {r['valid_slices']:>6} "
              f"{r['n_gt_lesions']:>5} {r['n_pred_lesions']:>6} "
              f"{r['lesion_TP']:>4} {r['lesion_FN']:>4} {r['lesion_FP']:>4} "
              f"{r['vessel_gt']:>5} {r['vessel_pred']:>6} {v_ok:>6}")

    # Save text results
    save_path = os.path.join(output_dir, "test_results.txt")
    with open(save_path, "w") as f:
        f.write("VESSEL-LEVEL\n")
        f.write(f"Total: {len(vessel_gt)}, GT positive: {sum(vessel_gt)}\n")
        f.write(f"CM:\n{cm_v}\n")
        for k, v in m_v.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nLESION-LEVEL\n")
        f.write(f"GT lesions: {l_TP+l_FN}, Pred lesions: {l_TP+l_FP}\n")
        f.write(f"TP={l_TP}  FP={l_FP}  FN={l_FN}  TN={l_TN}\n")
        f.write(f"CM:\n{cm_l}\n")
        for k, v in m_l.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nPER-SAMPLE\n")
        f.write(f"{'Name':<35} ValidZ  GT_L  Pred_L  TP  FN  FP  V_GT  V_Pred  V_OK\n")
        for r in per_sample:
            v_ok = "YES" if r['vessel_gt'] == r['vessel_pred'] else "NO"
            f.write(f"{r['name']:<35} {r['valid_slices']:>6} "
                    f"{r['n_gt_lesions']:>5} {r['n_pred_lesions']:>6} "
                    f"{r['lesion_TP']:>4} {r['lesion_FN']:>4} {r['lesion_FP']:>4} "
                    f"{r['vessel_gt']:>5} {r['vessel_pred']:>6}  {v_ok}\n")

    print(f"\nResults saved to : {save_path}")
    print(f"PNG images saved to: {vis_dir}/")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()