import os
import re
import torch
from tqdm import tqdm
from framework import sc_net_framework
from config import opt

# ── label mapping ─────────────────────────────────────────────
# Model output indices 0-6:
#   0         = background
#   1,2,3     = NS+NC, NS+M, NS+C  (non-significant)
#   4,5,6     = S+NC,  S+M,  S+C   (significant)
#
# Vessel-level classes:
#   0 = Normal         (no lesion predicted)
#   1 = Non-Significant (highest pred class is 1-3)
#   2 = Significant     (highest pred class is 4-6)

BG_IDX = 0


def get_vessel_gt(gt_labels):
    """
    gt_labels: 1-indexed labels (1-6) from targets dict.
    Returns: 0=Normal, 1=NS, 2=Significant.
    """
    if gt_labels.numel() == 0:
        return 0
    max_label = gt_labels.max().item()   # 1-6
    return (max_label - 1) // 3 + 1     # 1-3→1(NS), 4-6→2(S)


def get_vessel_pred(pred_logits_b, score_thresh):
    """
    pred_logits_b: (num_queries, 7) softmax probabilities from model in testing mode.
    Returns: 0=Normal, 1=NS, 2=Significant.
    Significant overrides NS — check all queries.
    """
    pred_classes = pred_logits_b.argmax(dim=-1)   # (num_queries,)
    pred_scores  = pred_logits_b.max(dim=-1).values

    vessel_pred = 0
    for q in range(len(pred_classes)):
        cls = pred_classes[q].item()
        sc  = pred_scores[q].item()
        if cls == BG_IDX or sc < score_thresh:
            continue
        if 4 <= cls <= 6:
            return 2   # significant — highest priority, return immediately
        elif 1 <= cls <= 3:
            vessel_pred = max(vessel_pred, 1)

    return vessel_pred


def evaluate_one(checkpoint_path, test_loader, device, score_thresh):
    # build model directly without framework to avoid dataloader rebuild
    fw = sc_net_framework(pattern='inference')   # 'inference' skips dataloader build
    model = fw.model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)

    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern   = 'testing'
    model.object_detection_framework.pattern = 'testing'

    correct = 0
    total   = 0
    per_cls_correct = [0, 0, 0]
    per_cls_total   = [0, 0, 0]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='  eval', leave=False):
            images, targets = batch[0], batch[1]
            images = images.to(device)
            od_outputs = model(images)
            pred_logits = od_outputs['pred_logits']

            for b in range(images.shape[0]):
                gt_labels = targets[b]['labels'].cpu()
                vessel_gt   = get_vessel_gt(gt_labels)
                vessel_pred = get_vessel_pred(pred_logits[b].cpu(), score_thresh)

                per_cls_total[vessel_gt] += 1
                if vessel_gt == vessel_pred:
                    correct += 1
                    per_cls_correct[vessel_gt] += 1
                total += 1

    acc = correct / total if total > 0 else 0.0
    per_cls_acc = [
        per_cls_correct[i] / per_cls_total[i] if per_cls_total[i] > 0 else 0.0
        for i in range(3)
    ]
    return acc, per_cls_acc, per_cls_total


def sweep(checkpoint_dir, name_prefix, device='cuda:0',
          score_thresh=0.05, out_file='sweep_results.txt'):

    # find all matching checkpoints
    pattern = re.compile(rf'^{re.escape(name_prefix)}_epoch(\d+)\.pth$')
    # in sweep(), replace the checkpoint sorting with:
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        m = pattern.match(f)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num % 5 == 0 or epoch_num == 1:  # every 5th + epoch 1
                checkpoints.append((epoch_num, os.path.join(checkpoint_dir, f)))
    checkpoints.sort(key=lambda x: x[0])

    # build test loader once — reuse across all checkpoints
    fw = sc_net_framework(pattern='fine_tuning')
    test_loader = fw.dataLoader_eval

    results = []
    for epoch, ckpt_path in tqdm(checkpoints, desc='Sweeping'):
        try:
            acc, per_cls_acc, per_cls_total = evaluate_one(
                ckpt_path, test_loader, device, score_thresh)
            results.append((epoch, ckpt_path, acc, per_cls_acc, per_cls_total))
            tqdm.write(
                f"  epoch {epoch:03d}: acc={acc:.4f} | "
                f"Normal={per_cls_acc[0]:.3f}({per_cls_total[0]}) "
                f"NS={per_cls_acc[1]:.3f}({per_cls_total[1]}) "
                f"Sig={per_cls_acc[2]:.3f}({per_cls_total[2]})"
            )
        except Exception as e:
            tqdm.write(f"  epoch {epoch:03d}: FAILED — {e}")

    # sort by overall accuracy
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    with open(out_file, 'w') as f:
        f.write(f"Sweep: {name_prefix}\n")
        f.write(f"Score threshold: {score_thresh}\n\n")
        f.write(f"{'Rank':<5} {'Epoch':<7} {'Acc':<8} "
                f"{'Normal':<10} {'NS':<10} {'Sig':<10} "
                f"{'N_norm':<8} {'N_ns':<8} {'N_sig':<8}\n")
        f.write("-" * 80 + "\n")
        for rank, (epoch, path, acc, pca, pct) in enumerate(results_sorted):
            f.write(
                f"{rank+1:<5} {epoch:<7} {acc:<8.4f} "
                f"{pca[0]:<10.4f} {pca[1]:<10.4f} {pca[2]:<10.4f} "
                f"{pct[0]:<8} {pct[1]:<8} {pct[2]:<8}\n"
            )

        f.write("\n--- Chronological order ---\n")
        for epoch, path, acc, pca, pct in results:
            f.write(f"epoch {epoch:03d}: acc={acc:.4f} | "
                    f"Normal={pca[0]:.3f} NS={pca[1]:.3f} Sig={pca[2]:.3f}\n")

    print(f"\nTop 5 by accuracy:")
    for rank, (epoch, path, acc, pca, pct) in enumerate(results_sorted[:5]):
        print(f"  {rank+1}. epoch {epoch:03d}: acc={acc:.4f} | "
              f"Normal={pca[0]:.3f} NS={pca[1]:.3f} Sig={pca[2]:.3f}")
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    sweep(
        checkpoint_dir='/home/joshua/CAD_diagnosis-master',
        name_prefix='model_32x25x8_eos0.02',
        device='cuda:0',
        score_thresh=0.05,
        out_file='sweep_model_32x25x8_eos0.02.txt',
    )
