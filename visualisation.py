import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

from framework import sc_net_framework
from config_1 import opt

# ── label definitions ────────────────────────────────────────
CLASS_NAMES = ['NS+NC', 'NS+M', 'NS+C', 'S+NC', 'S+M', 'S+C']
BG_IDX = 6  # num_classes index = background

# colour per class for the classification bars
CLASS_COLORS = {
    0: [0.25, 0.41, 0.88],   # NS+NC  royalblue
    1: [0.39, 0.58, 0.93],   # NS+M   cornflowerblue
    2: [0.27, 0.51, 0.71],   # NS+C   steelblue
    3: [1.00, 0.27, 0.00],   # S+NC   orangered
    4: [1.00, 0.55, 0.00],   # S+M    darkorange
    5: [0.55, 0.00, 0.00],   # S+C    darkred
}


# ── utility ──────────────────────────────────────────────────


def get_labeled_segments(label_array):
    """Contiguous runs of SAME non-background label -> list of (start, end, label)."""
    segments, in_seg, current_label, start = [], False, 0, BG_IDX
    for i, v in enumerate(label_array):
        v = int(v)
        if v != BG_IDX and not in_seg:  # Skip background (class 6)
            start, in_seg, current_label = i, True, v
        elif v != BG_IDX and in_seg and v != current_label:
            segments.append((start, i - 1, current_label))
            start, current_label = i, v
        elif v == BG_IDX and in_seg:  # End segment if background is encountered
            segments.append((start, i - 1, current_label))
            in_seg = False
    if in_seg:
        segments.append((start, len(label_array) - 1, current_label))
    return segments


def get_contiguous_segments(binary_array):
    """Contiguous runs of 1s -> list of (start, end)."""
    segments, in_seg = [], False
    for i, v in enumerate(binary_array):
        if v == 1 and not in_seg:
            start, in_seg = i, True
        elif v == BG_IDX and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary_array) - 1))
    return segments


def segments_overlap(seg, pred_array):
    return pred_array[seg[0]:seg[1] + 1].max() == 1


def compute_metrics(tp, fp, fn, tn):
    precision    = tp / (tp + fp + 1e-9)
    recall       = tp / (tp + fn + 1e-9)
    specificity  = tn / (tn + fp + 1e-9)
    f1           = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy     = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=precision, recall=recall,
                specificity=specificity, f1=f1, accuracy=accuracy)


# ── boxes -> per-slice label array ───────────────────────────

def boxes_to_slice_labels(pred_logits, pred_boxes, volume_depth, score_thresh):
    """
    Convert model output (boxes + logits) to a per-slice label array
    of length volume_depth (0 = background).
    Each query votes on a slice range; highest-scoring non-bg wins per slice.
    """
    slice_labels = np.full(volume_depth, BG_IDX, dtype=np.int32)
    slice_scores = np.zeros(volume_depth, dtype=np.float32)

    scores, labels = pred_logits.max(dim=-1)   # (num_queries,)
    for q in range(len(labels)):
        lbl = labels[q].item()
        sc  = scores[q].item()
        if lbl == BG_IDX or sc < score_thresh:
            continue
        start_idx = int(pred_boxes[q, 0].item() * volume_depth)
        end_idx   = int(pred_boxes[q, 1].item() * volume_depth)
        start_idx = max(0, min(start_idx, volume_depth - 1))
        end_idx   = max(start_idx, min(end_idx, volume_depth - 1))
        for z in range(start_idx, end_idx + 1):
            if sc > slice_scores[z]:
                slice_scores[z] = sc
                slice_labels[z] = lbl

    return slice_labels


# ── visualisation ─────────────────────────────────────────────

def make_label_bar(label_array, length):
    """
    RGB bar image (bar_h x length x 3).
    black = normal, coloured = class colour.
    """
    bar_h = 30
    img = np.zeros((bar_h, length, 3), dtype=np.float32)
    for z in range(length):
        lbl = int(label_array[z])
        if lbl != BG_IDX:
            img[:, z, :] = CLASS_COLORS.get(lbl, [0.5, 0.5, 0.5])
    return img


def save_visualization(vol_np, gt_label_array, pred_label_array,
                       gt_segs, pred_segs, valid_len, name, save_dir):
    """
    3-row figure matching reference style:
      Row 1 – CPR longitudinal strip (greyscale)
      Row 2 – GT  coloured bar
      Row 3 – Pred coloured bar
    """
    Z, H, W = vol_np.shape
    cw = W // 2

    longit = vol_np[:valid_len, :, cw].T           # (H, valid_len)
    vmin   = np.percentile(longit, 1)
    vmax   = np.percentile(longit, 99)
    longit_norm = np.clip((longit - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # per-sample stats for title
    v_gt   = int((gt_label_array[:valid_len]   < BG_IDX).min()) if valid_len > 0 else 0
    v_pred = int((pred_label_array[:valid_len] < BG_IDX).min()) if valid_len > 0 else 0
    v_ok   = "✓" if v_gt == v_pred else "✗"

    n_lesions  = len(gt_segs)
    n_detected = sum(1 for s in gt_segs
                     if segments_overlap(s, (pred_label_array < BG_IDX).astype(np.int32)))
    n_missed   = n_lesions - n_detected
    n_fp       = sum(1 for ps in pred_segs
                     if not any(not (ps[1] < gs[0] or ps[0] > gs[1])
                                for gs in gt_segs))

    title = (f"{name}  |  vessel gt={v_gt} pred={v_pred} {v_ok}  |  "
             f"lesions={n_lesions} detected={n_detected} "
             f"missed={n_missed} fp={n_fp}")

    bar_gt   = make_label_bar(gt_label_array,   valid_len)
    bar_pred = make_label_bar(pred_label_array, valid_len)

    fig_w = max(14, valid_len // 8)
    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_w, 7),
        gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.35}
    )
    fig.suptitle(title, fontsize=9, y=1.01)

    imshow_kw = dict(aspect="auto", origin="upper", interpolation="nearest")

    axes[0].imshow(longit_norm, cmap="gray", **imshow_kw)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    tick_pos = np.arange(0, valid_len, max(1, valid_len // 6)).tolist()

    axes[1].imshow(bar_gt, **imshow_kw)
    axes[1].set_ylabel("GT",   fontsize=9, labelpad=4, va="center")
    axes[1].set_yticks([])
    axes[1].set_xticks(tick_pos)
    axes[1].tick_params(axis="x", labelsize=8)

    axes[2].imshow(bar_pred, **imshow_kw)
    axes[2].set_ylabel("Pred", fontsize=9, labelpad=4, va="center")
    axes[2].set_yticks([])
    axes[2].set_xticks(tick_pos)
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].set_xlabel("Z (slice index)", fontsize=9)

    # shared legend
    legend_els = [mpatches.Patch(facecolor='black', label='normal')] + \
                 [mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
                  for i in range(len(CLASS_NAMES))]
    axes[1].legend(handles=legend_els, loc="upper right",
                   fontsize=7, framealpha=0.8, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── model loading ─────────────────────────────────────────────

def load_model(checkpoint_path, device):
    fw = sc_net_framework(pattern='fine_tuning')
    model = fw.model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern  = 'testing'
    model.object_detection_framework.pattern = 'testing'
    return model, fw.dataLoader_eval


# ── main evaluate loop ────────────────────────────────────────

def evaluate(checkpoint_path, out_dir='eval_results', device='cuda',
             score_thresh=0.05, iou_thresh=0.3):

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model, test_loader = load_model(checkpoint_path, device)
    volume_depth = opt.net_params["input_shape"][0]

    # accumulators
    vessel_gt_list, vessel_pred_list = [], []
    lesion_TP, lesion_FN, lesion_FP, lesion_TN = 0, 0, 0, 0
    all_pred_cls, all_gt_cls = [], []
    per_sample = []

    sample_idx = 0
    for images, targets, names in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        with torch.no_grad():
            od_outputs = model(images)

        pred_logits = od_outputs['pred_logits']   # (B, num_queries, num_classes+1)
        pred_boxes  = od_outputs['pred_boxes']    # (B, num_queries, 2)

        for b in range(images.shape[0]):
            name = names[b]

            gt_labels = targets[b]['labels'].cpu()   # (num_gt,) 0-indexed classes
            gt_boxes  = targets[b]['boxes'].cpu()    # (num_gt, 2) normalised

            # build per-slice GT label array from boxes
            gt_label_array = np.full(volume_depth, BG_IDX, dtype=np.int32)
            for k in range(len(gt_labels)):
                lbl = gt_labels[k].item()
                s = int(gt_boxes[k, 0].item() * volume_depth)
                e = int(gt_boxes[k, 1].item() * volume_depth)
                s = max(0, min(s, volume_depth - 1))
                e = max(s, min(e, volume_depth - 1))
                gt_label_array[s:e + 1] = lbl

            # build per-slice pred label array from model output
            pred_label_array = boxes_to_slice_labels(
                pred_logits[b].cpu(), pred_boxes[b].cpu(),
                volume_depth, score_thresh)

            valid_len = volume_depth   # already fixed shape from dataloader
            assert valid_len > 0
            gt_segs   = get_labeled_segments(gt_label_array[:valid_len])
            pred_segs = get_contiguous_segments(
                (pred_label_array[:valid_len] < BG_IDX).astype(np.int32))

            # vessel-level
            v_gt   = int((gt_label_array[:valid_len]   < BG_IDX).max()) 
            v_pred = int((pred_label_array[:valid_len] < BG_IDX).max()) 
            vessel_gt_list.append(v_gt)
            vessel_pred_list.append(v_pred)

            # lesion-instance-level
            vol_TP, vol_FN, vol_FP = 0, 0, 0
            if v_gt == BG_IDX and v_pred == BG_IDX:
                lesion_TN += 1
            for seg in gt_segs:
                if segments_overlap(seg, (pred_label_array < BG_IDX).astype(np.int32)):
                    lesion_TP += 1; vol_TP += 1
                else:
                    lesion_FN += 1; vol_FN += 1
            for pseg in pred_segs:
                if not any(not (pseg[1] < gs[0] or pseg[0] > gs[1]) for gs in gt_segs):
                    lesion_FP += 1; vol_FP += 1

            # classification accuracy on matched slices
            for z in range(valid_len):
                gt_lbl = int(gt_label_array[z])
                pr_lbl = int(pred_label_array[z])
                if gt_lbl != BG_IDX:
                    all_gt_cls.append(gt_lbl)
                    all_pred_cls.append(pr_lbl)

            per_sample.append({
                'name':           name,
                'valid_slices':   valid_len,
                'n_gt_lesions':   len(gt_segs),
                'n_pred_lesions': len(pred_segs),
                'lesion_TP': vol_TP, 'lesion_FN': vol_FN, 'lesion_FP': vol_FP,
                'vessel_gt': v_gt,   'vessel_pred': v_pred,
            })

            save_visualization(
                vol_np           = images[b].cpu().numpy(),
                gt_label_array   = gt_label_array,
                pred_label_array = pred_label_array,
                gt_segs          = gt_segs,
                pred_segs        = pred_segs,
                valid_len        = valid_len,
                name             = name,
                save_dir         = vis_dir,
            )
            sample_idx += 1

    # ── vessel-level confusion matrix ────────────────────────
    cm_v = confusion_matrix(vessel_gt_list, vessel_pred_list, labels=[0, 1])
    m_v  = compute_metrics(int(cm_v[1,1]), int(cm_v[0,1]),
                            int(cm_v[1,0]), int(cm_v[0,0]))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_v, display_labels=['Normal', 'Lesion']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Vessel-Level Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_vessel.png'), dpi=150)
    plt.close()

    # ── per-class confusion matrix (slice level) ─────────────
    if len(all_gt_cls) > 0:
        cm_cls = confusion_matrix(all_gt_cls, all_pred_cls,
                                  labels=list(range(len(CLASS_NAMES) + 1)))
        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay(
            cm_cls,
            display_labels=CLASS_NAMES + ['bg']
        ).plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title('Per-Class Confusion Matrix (lesion slices)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'confusion_matrix_class.png'), dpi=150)
        plt.close()

        cls_report = classification_report(
            all_gt_cls, all_pred_cls,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, zero_division=0)
    else:
        cls_report = "No lesion slices found."

    # ── write results txt ────────────────────────────────────
    stats_path = os.path.join(out_dir, 'test_results.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Checkpoint : {checkpoint_path}\n")
        f.write(f"Score thresh: {score_thresh}  IoU thresh: {iou_thresh}\n\n")

        f.write("VESSEL-LEVEL\n")
        f.write(f"  Total vessels  : {len(vessel_gt_list)}\n")
        f.write(f"  GT positive    : {sum(vessel_gt_list)}\n")
        f.write(f"  TP={m_v['TP']}  FP={m_v['FP']}  FN={m_v['FN']}  TN={m_v['TN']}\n")
        for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
            f.write(f"  {k:<20}: {m_v[k]:.4f}\n")

        f.write("\nLESION-LEVEL\n")
        l_m = compute_metrics(lesion_TP, lesion_FP, lesion_FN, lesion_TN)
        f.write(f"  GT lesions  : {lesion_TP + lesion_FN}\n")
        f.write(f"  Pred lesions: {lesion_TP + lesion_FP}\n")
        f.write(f"  TP={l_m['TP']}  FP={l_m['FP']}  FN={l_m['FN']}  TN={l_m['TN']}\n")
        for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
            f.write(f"  {k:<20}: {l_m[k]:.4f}\n")

        f.write("\nPER-CLASS CLASSIFICATION (lesion slices)\n")
        f.write(cls_report)

        f.write("\nPER-SAMPLE\n")
        f.write(f"{'Name':<20} {'ValidZ':>6} {'GT_L':>5} {'Pred_L':>6} "
                f"{'TP':>4} {'FN':>4} {'FP':>4} {'V_GT':>5} {'V_Pred':>6} {'V_OK':>6}\n")
        for r in per_sample:
            v_ok = "YES" if r['vessel_gt'] == r['vessel_pred'] else "NO"
            f.write(f"{r['name']:<20} {r['valid_slices']:>6} "
                    f"{r['n_gt_lesions']:>5} {r['n_pred_lesions']:>6} "
                    f"{r['lesion_TP']:>4} {r['lesion_FN']:>4} {r['lesion_FP']:>4} "
                    f"{r['vessel_gt']:>5} {r['vessel_pred']:>6}  {v_ok}\n")

    print(f"\nVessel-level: P:{m_v['precision']:.4f} R:{m_v['recall']:.4f} "
          f"F1:{m_v['f1']:.4f} Acc:{m_v['accuracy']:.4f}")
    print(f"Results saved to : {stats_path}")
    print(f"PNGs saved to    : {vis_dir}/")


if __name__ == '__main__':
    evaluate(
        checkpoint_path='/home/joshua/CAD_diagnosis-master/model_58x40x8_unchanged_labels_epoch026.pth',
        out_dir='eval_results_unchanged_labels_epoch026_no_shift',
        device='cuda:0',
        score_thresh=0.01,
        iou_thresh=0.1,
    )

