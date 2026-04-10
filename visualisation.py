import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from framework import sc_net_framework
from config import opt

# ── label definitions ────────────────────────────────────────
CLASS_NAMES = ['Normal', 'Non-Significant', 'Significant']
BG_IDX = 0 

CLASS_COLORS = {
    1: [0.25, 0.41, 0.88],   # NS+NC  royalblue
    2: [0.39, 0.58, 0.93],   # NS+M   cornflowerblue
    3: [0.27, 0.51, 0.71],   # NS+C   steelblue
    4: [1.00, 0.27, 0.00],   # S+NC   orangered
    5: [1.00, 0.55, 0.00],   # S+M    darkorange
    6: [0.55, 0.00, 0.00],   # S+C    darkred
}

# ── utility ──────────────────────────────────────────────────

def compute_metrics(cm):
    """
    Compute precision, recall, f1, and accuracy from a confusion matrix.
    This function works with a 3x3 confusion matrix for vessel-level classification.
    """
    # Extract confusion matrix components
    tp = np.diagonal(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (tp + fp + fn)

    # Calculate metrics for each class (Normal, Non-Significant, Significant)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = np.sum(tp) / np.sum(cm)

    # Return all metrics for each class
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': accuracy
    } 

# ── visualization helpers ────────────────────────────────────

def boxes_to_slice_label_array(boxes, labels, z_len, scores=None, score_thresh=0.0, bg_idx=BG_IDX):
    """
    Convert 1D normalized boxes + labels into a per-slice label array.

    Assumes boxes are in (start, end) format, normalized to [0, 1].
    """
    arr = np.full(z_len, bg_idx, dtype=np.int32)

    if boxes is None or len(boxes) == 0:
        return arr

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    for i in range(len(boxes)):
        if scores is not None and scores[i] < score_thresh:
            continue

        cls = int(labels[i])
        if cls == bg_idx:
            continue

        start = int(round(boxes[i][0] * z_len))
        end   = int(round(boxes[i][1] * z_len)) - 1
        start = max(0, min(z_len - 1, start))
        end   = max(0, min(z_len - 1, end))

        if end < start:
            continue

        # overwrite slice range with this class
        arr[start:end + 1] = cls

    return arr


def make_label_bar(label_array, length):
    """Create a colored label bar for visualization."""
    bar_h = 30
    img = np.full((bar_h, length, 3), 0, dtype=np.float32)
    for z in range(length):
        lbl = int(label_array[z])
        if lbl != BG_IDX:
            img[:, z, :] = CLASS_COLORS.get(lbl, [0.5, 0.5, 0.5])
    return img

def save_visualization(vol_np, gt_label_array, pred_label_array, name, save_dir):
    # handle possible channel dim
    vol_np = np.asarray(vol_np)
    vol_np = np.squeeze(vol_np)

    if vol_np.ndim != 3:
        raise ValueError(f"Expected volume with shape (Z, H, W), got {vol_np.shape}")

    Z, H, W = vol_np.shape
    cw = W // 2

    longit = vol_np[:, :, cw].T
    vmin = np.percentile(longit, 1)
    vmax = np.percentile(longit, 99)
    longit_norm = np.clip((longit - vmin) / (vmax - vmin + 1e-6), 0, 1)

    bar_gt   = make_label_bar(gt_label_array, Z)
    bar_pred = make_label_bar(pred_label_array, Z)

    fig_w = max(14, Z // 8)
    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_w, 7),
        gridspec_kw={"height_ratios": [4, 0.6, 0.6], "hspace": 0.25}
    )

    axes[0].imshow(longit_norm, cmap="gray", aspect="auto", origin="upper", interpolation="nearest")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(name, fontsize=10)

    axes[1].imshow(bar_gt, aspect="auto", origin="upper", interpolation="nearest")
    axes[1].set_ylabel("GT", fontsize=9, labelpad=8, va="center")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(bar_pred, aspect="auto", origin="upper", interpolation="nearest")
    axes[2].set_ylabel("Pred", fontsize=9, labelpad=8, va="center")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Z (slice index)", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)




# ── model loading ─────────────────────────────────────────────

def load_model(checkpoint_path, device, cfg):
    fw = sc_net_framework(pattern='fine_tuning', cfg=cfg)
    model = fw.model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern   = 'testing'
    model.object_detection_framework.pattern = 'testing'
    return model, fw.dataLoader_eval


# ── main evaluate loop ────────────────────────────────────────

def evaluate(checkpoint_path, out_dir='eval_results', device='cuda', score_thresh=0.05, iou_thresh=0.5):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model, test_loader = load_model(checkpoint_path, device, opt)

    vessel_gt_list, vessel_pred_list = [], []

    sample_idx = 0
    for images, targets, names in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        with torch.no_grad():
            od_outputs = model(images)

        pred_logits = od_outputs['pred_logits']   # (B, num_queries, 7)
        pred_boxes  = od_outputs['pred_boxes']    # (B, num_queries, 2)

        for b in range(images.shape[0]):
            name = names[b]

            gt_labels = targets[b]['labels'].cpu()   # (num_gt,) labels 1-6
            gt_boxes  = targets[b]['boxes'].cpu()    # (num_gt, 2) normalised [0,1]

            # Vessel-level — Assign highest severity lesion class to vessel
            probs = pred_logits[b].softmax(dim=-1)              # (num_queries, 7)
            pred_classes = pred_logits[b].argmax(dim=-1)        # class over all 7 classes
            prob_scores = probs[:, :].max(dim=-1).values      

            # Assigning the vessel class based on the highest severity lesion
            if gt_labels.numel() == 0:
                vessel_gt = 0  # no lesions, so vessel is normal
            else:
                vessel_gt = (gt_labels.max().item() - 1) // 3 + 1 #map 1-3 to 1, 4-6 to 2
            
            vessel_pred = 0  # Default: normal vessel (no lesion)

            pairs = sorted(
                zip(pred_classes.tolist(), prob_scores.tolist()),
                key=lambda x: (x[0], x[1]),   # class first, then score
                reverse=True
            )

            for cls, sc in pairs:

                if sc < score_thresh:
                    continue

                vessel_pred = (cls - 1) // 3 + 1 #map 1-3 to 1, 4-6 to 2, 0-0 works cos -1//3 = -1
                break

            vessel_gt_list.append(vessel_gt)
            vessel_pred_list.append(vessel_pred)


            z_len = int(np.squeeze(images[b].cpu().numpy()).shape[0])

            gt_slice_labels = boxes_to_slice_label_array(
                boxes=gt_boxes,
                labels=gt_labels,
                z_len=z_len,
                scores=None,
                score_thresh=0.0,
            )

            pred_slice_labels = boxes_to_slice_label_array(
                boxes=pred_boxes[b],
                labels=pred_classes,
                z_len=z_len,
                scores=prob_scores,
                score_thresh=score_thresh,
            )

            save_visualization(
                vol_np=images[b].cpu().numpy(),
                gt_label_array=gt_slice_labels,
                pred_label_array=pred_slice_labels,
                name=name,
                save_dir=vis_dir,
            )
            sample_idx += 1

    # ── vessel-level confusion matrix ────────────────────────
    cm_v = confusion_matrix(vessel_gt_list, vessel_pred_list, labels=[0, 1, 2])  # 0: normal, 1: non-significant, 2: significant

    metrics = compute_metrics(cm_v)


    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_v, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Vessel-Level Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_vessel.png'), dpi=150)
    plt.close()

    # write stats
    stats_path = os.path.join(out_dir, 'vessel_results.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Score thresh: {score_thresh}\n\n")
        f.write(f"Overall accuracy: {metrics['accuracy']:.4f}\n\n")
        for i, cls in enumerate(CLASS_NAMES):
            f.write(f"{cls}:\n")
            f.write(f"  precision  : {metrics['precision'][i]:.4f}\n")
            f.write(f"  recall     : {metrics['recall'][i]:.4f}\n")
            f.write(f"  specificity: {metrics['specificity'][i]:.4f}\n")
            f.write(f"  f1         : {metrics['f1'][i]:.4f}\n")


    print(f"\nVessel-level accuracy: {metrics['accuracy']:.4f}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: P={metrics['precision'][i]:.4f} R={metrics['recall'][i]:.4f} F1={metrics['f1'][i]:.4f}")
    print(f"Results saved to : {out_dir}")
    print(f"PNGs saved to    : {vis_dir}/")


if __name__ == '__main__':
    evaluate(
        checkpoint_path='/home/joshua/CAD_diagnosis-master/model_60x40x8_eos_1/model_60x40x8_EOS_1_best.pth',
        out_dir='eval_results_60x40x8_eos_1',
        device='cuda:0',
        score_thresh=0.05,
        iou_thresh=0.1,
    )