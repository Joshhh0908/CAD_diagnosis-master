import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from framework import sc_net_framework
from config_1 import opt

# ── label definitions ────────────────────────────────────────
CLASS_NAMES = ['Normal', 'Non-Significant', 'Significant']
BG_IDX = 6  # background is last index in model output space

CLASS_COLORS = {
    0: [0.25, 0.41, 0.88],   # NS+NC  royalblue
    1: [0.39, 0.58, 0.93],   # NS+M   cornflowerblue
    2: [0.27, 0.51, 0.71],   # NS+C   steelblue
    3: [1.00, 0.27, 0.00],   # S+NC   orangered
    4: [1.00, 0.55, 0.00],   # S+M    darkorange
    5: [0.55, 0.00, 0.00],   # S+C    darkred
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

def save_visualization(vol_np, gt_label_array, pred_label_array, name, save_dir):
    Z, H, W = vol_np.shape
    cw = W // 2

    longit = vol_np[:, :, cw].T  # Only a single slice here (as we are focusing on vessel-level)
    vmin   = np.percentile(longit, 1)
    vmax   = np.percentile(longit, 99)
    longit_norm = np.clip((longit - vmin) / (vmax - vmin + 1e-6), 0, 1)

    bar_gt   = make_label_bar(gt_label_array, longit.shape[1])
    bar_pred = make_label_bar(pred_label_array, longit.shape[1])

    fig_w = max(14, longit.shape[1] // 8)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_w, 7),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35}
    )
    axes[0].imshow(longit_norm, cmap="gray", aspect="auto", origin="upper", interpolation="nearest")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(bar_gt, aspect="auto", origin="upper", interpolation="nearest")
    axes[1].set_ylabel("GT", fontsize=9, labelpad=4, va="center")
    axes[1].set_yticks([])

    axes[2].imshow(bar_pred, aspect="auto", origin="upper", interpolation="nearest")
    axes[2].set_ylabel("Pred", fontsize=9, labelpad=4, va="center")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Z (slice index)", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

def make_label_bar(label_array, length):
    """Create a colored label bar for visualization."""
    bar_h = 30
    img = np.zeros((bar_h, length, 3), dtype=np.float32)
    for z in range(length):
        lbl = int(label_array[z])
        if lbl != BG_IDX:
            img[:, z, :] = CLASS_COLORS.get(lbl, [0.5, 0.5, 0.5])
    return img


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
    model.sampling_point_framework.pattern   = 'testing'
    model.object_detection_framework.pattern = 'testing'
    return model, fw.dataLoader_eval


# ── main evaluate loop ────────────────────────────────────────

def evaluate(checkpoint_path, out_dir='eval_results', device='cuda', score_thresh=0.05, iou_thresh=0.5):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model, test_loader = load_model(checkpoint_path, device)

    vessel_gt_list, vessel_pred_list = [], []

    sample_idx = 0
    for images, targets in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        with torch.no_grad():
            od_outputs = model(images)

        pred_logits = od_outputs['pred_logits']   # (B, num_queries, 7)
        pred_boxes  = od_outputs['pred_boxes']    # (B, num_queries, 2)

        for b in range(images.shape[0]):
            name = f"sample_{sample_idx:04d}"

            gt_labels = targets[b]['labels'].cpu()   # (num_gt,) 0-indexed classes 0-5
            gt_boxes  = targets[b]['boxes'].cpu()    # (num_gt, 2) normalised [0,1]

            # Vessel-level — Assign highest severity lesion class to vessel
            pred_classes = pred_logits[b].max(dim=-1)[1]  # Predicted classes
            pred_scores = pred_logits[b].max(dim=-1)[0]   # Predicted scores

            # Assigning the vessel class based on the highest severity lesion
            if gt_labels.numel() == 0:
                vessel_gt = 0  # No lesions, so normal vessel
            else:
                vessel_gt = (max(gt_labels) // 3) + 1  # 0: Normal (No lesions), 1: Non-Significant (0-2), 2: Significant (3-5)
            vessel_pred = 0  # Default: normal vessel (no lesion)

            if pred_scores.max() > score_thresh:  # If there is a lesion
                highest_pred_class = pred_classes.max().item()
                if 0 <= highest_pred_class <= 2:
                    vessel_pred = 1  # Non-significant
                elif 3 <= highest_pred_class <= 5:
                    vessel_pred = 2  # Significant

            vessel_gt_list.append(vessel_gt)
            vessel_pred_list.append(vessel_pred)

            # save_visualization(
            #     vol_np=images[b].cpu().numpy(),
            #     gt_label_array=gt_labels,
            #     pred_label_array=pred_classes,
            #     name=name,
            #     save_dir=vis_dir,
            # )
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

    print(f"\nVessel-level: P:{m_v['precision']:.4f} R:{m_v['recall']:.4f} "
          f"F1:{m_v['f1']:.4f} Acc:{m_v['accuracy']:.4f}")
    print(f"Results saved to : {out_dir}")
    print(f"PNGs saved to    : {vis_dir}/")


if __name__ == '__main__':
    evaluate(
        checkpoint_path='/home/joshua/CAD_diagnosis-master/model_58x40x8/model_58x40x8_unchanged_labels_epoch026.pth',
        out_dir='eval_results_vessel_only',
        device='cuda:0',
        score_thresh=0.01,
        iou_thresh=0.1,
    )