import torch
from tqdm import tqdm
from functions import boxes_cw_to_se


JOINT_TO_STEN   = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
JOINT_TO_PLAQUE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}

BG_IDX = 0

def get_vessel_pred(pred_logits_b, score_thresh):
    """
    pred_logits_b: (num_queries, 7) softmax probabilities from model in testing mode.
    Returns: 0=Normal, 1=NS, 2=Significant.
    Significant overrides NS — check all queries.
    """
    pred_classes = pred_logits_b.argmax(dim=-1)   # (num_queries,)
    pred_scores  = pred_logits_b.max(dim=-1).values
    vessel_pred = 0

    pairs = sorted(
                zip(pred_classes.tolist(), pred_scores.tolist()),
                key=lambda x: (x[0], x[1]),   # class first, then score
                reverse=True
            )
    for cls, sc in pairs:
        if sc < score_thresh:
            continue

        vessel_pred = JOINT_TO_STEN[cls]
        break

    return vessel_pred

def od_inference(od_outputs, targets, eval_length,score_thresh=0.05):

    correct = 0
    total = 0
    per_cls_correct = [0, 0, 0]
    per_cls_total   = [0, 0, 0]
    pred_logits = od_outputs['pred_logits']

    for b in range(eval_length):
        gt_labels = targets[b]['labels'].cpu()
        # GT vessel class
        if gt_labels.numel() == 0:
            vessel_gt = 0  # no lesions, so vessel is normal
        else:
            vessel_gt = JOINT_TO_STEN[gt_labels.max().item()]#map 1-3 to 1, 4-6 to 2
            
        vessel_pred = get_vessel_pred(pred_logits[b].cpu(), score_thresh)

        per_cls_total[vessel_gt] += 1
        if vessel_gt == vessel_pred:
            correct += 1
            per_cls_correct[vessel_gt] += 1
        total += 1
    
    return correct, total

def sc_inference(sc_outputs, sc_targets):

    sc_logits = sc_outputs["pred_logits"]  # [B, L, 7]
    device = sc_logits.device

    # mappings
    sten_mapping = torch.tensor([JOINT_TO_STEN[i] for i in range(7)], device=device, dtype=torch.long)
    plaq_mapping = torch.tensor([JOINT_TO_PLAQUE[i] for i in range(7)], device=device, dtype=torch.long)

    # targets
    # print("sc_targets:", sc_targets)
    gt_seq = torch.stack([t['labels'].to(device) for t in sc_targets], dim=0)

    # predictions
    pred_seq = sc_logits.argmax(dim=-1)  # [B, L]

    # mapped classes
    pred_sten = sten_mapping[pred_seq]
    gt_sten   = sten_mapping[gt_seq]

    pred_plaq = plaq_mapping[pred_seq]
    gt_plaq   = plaq_mapping[gt_seq]

    # ---------------- cube-level ----------------

    cube_joint_correct = (pred_seq == gt_seq).sum().item()
    cube_joint_total   = gt_seq.numel()

    cube_sten_correct = (pred_sten == gt_sten).sum().item()
    cube_sten_total   = gt_sten.numel()

    cube_plaq_correct = (pred_plaq == gt_plaq).sum().item()
    cube_plaq_total   = gt_plaq.numel()

    # ---------------- vessel-level ----------------

    gt_vessel   = sten_mapping[gt_seq.max(dim=1).values]     # [B]
    pred_vessel = sten_mapping[pred_seq.max(dim=1).values]   # [B]

    vessel_sten_correct = (gt_vessel == pred_vessel).sum().item()
    vessel_total        = gt_vessel.size(0)

    # ---------------- per-class (vectorised) ----------------

    per_cube_sten_correct = []
    per_cube_sten_total   = []

    for cls in range(3):
        mask = (gt_sten == cls)
        per_cube_sten_total.append(mask.sum().item())
        per_cube_sten_correct.append(((pred_sten == gt_sten) & mask).sum().item())

    per_cube_plaq_correct = []
    per_cube_plaq_total   = []

    for cls in range(4):
        mask = (gt_plaq == cls)
        per_cube_plaq_total.append(mask.sum().item())
        per_cube_plaq_correct.append(((pred_plaq == gt_plaq) & mask).sum().item())

    # ---------------- return ----------------

    return {
        "cube_joint_correct": cube_joint_correct,
        "cube_joint_total": cube_joint_total,

        "cube_sten_correct": cube_sten_correct,
        "cube_sten_total": cube_sten_total,

        "cube_plaq_correct": cube_plaq_correct,
        "cube_plaq_total": cube_plaq_total,

        "cube_sten_per_class": per_cube_sten_correct,
        "cube_sten_per_class_total": per_cube_sten_total,

        "cube_plaq_per_class": per_cube_plaq_correct,
        "cube_plaq_per_class_total": per_cube_plaq_total,

        "vessel_sten_correct": vessel_sten_correct,
        "vessel_total": vessel_total,
    }

def eval_epoch(model, loss_fn, eval_loader, device, epoch, num_epochs):
    # ----------------EVALUATION----------------
    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern = 'testing'
    model.object_detection_framework.pattern = 'testing'
    val_sten_correct = [0, 0, 0]
    val_sten_total   = [0, 0, 0]

    val_plaq_correct = [0, 0, 0, 0]
    val_plaq_total   = [0, 0, 0, 0]

    val_loss = 0.0
    val_sc_loss = 0.0
    val_od_loss = 0.0
    val_dc_loss = 0.0
    val_box_loss = 0.0
    val_label_loss = 0.0

    val_od_correct = 0
    val_od_total = 0

    val_sc_cube_joint = 0
    val_sc_cube_sten  = 0
    val_sc_cube_plaq  = 0
    val_sc_vessel_sten  = 0

    val_sc_cube_total = 0
    val_sc_vessel_total = 0

    val_bar = tqdm(eval_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ",
                leave=False)
    
    with torch.no_grad():
        for images, od_targets, sc_targets, _ in val_bar:
            images = images.to(device)
            od_targets = [{k: v.to(device) for k, v in t.items()} for t in od_targets]
            sc_targets = [{k: v.to(device) for k, v in t.items()} for t in sc_targets]
            od_outputs, sc_outputs = model(images)
            od_outputs_for_loss = dict(od_outputs)
            od_outputs_for_loss["pred_boxes"] = boxes_cw_to_se(od_outputs["pred_boxes"])

            # box checking debug
            # boxes = od_outputs["pred_boxes"].reshape(-1, 2)
            loss, sc_loss, od_loss, dc_loss, loss_labels, loss_boxes = loss_fn(od_outputs_for_loss, sc_outputs, od_targets, sc_targets)

            val_loss += loss.item()
            val_sc_loss += sc_loss.item()
            val_od_loss += od_loss.item()
            val_dc_loss += dc_loss.item()
            val_label_loss += loss_labels.item()
            val_box_loss += loss_boxes.item()

            batch_size = images.size(0)
            L = sc_outputs["pred_logits"].shape[1]
            val_sc_cube_total += batch_size * L
            val_sc_vessel_total += batch_size

            correct, total = od_inference(od_outputs, od_targets, batch_size)
            val_od_correct += correct
            val_od_total += total

            sc_metrics = sc_inference(sc_outputs, sc_targets)

            val_sc_cube_joint += sc_metrics["cube_joint_correct"]
            val_sc_cube_sten  += sc_metrics["cube_sten_correct"]
            val_sc_cube_plaq  += sc_metrics["cube_plaq_correct"]
            val_sc_vessel_sten  += sc_metrics["vessel_sten_correct"]

            for i in range(3):
                val_sten_correct[i] += sc_metrics["cube_sten_per_class"][i]
                val_sten_total[i]   += sc_metrics["cube_sten_per_class_total"][i]

            for i in range(4):
                val_plaq_correct[i] += sc_metrics["cube_plaq_per_class"][i]
                val_plaq_total[i]   += sc_metrics["cube_plaq_per_class_total"][i]

            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss /= len(eval_loader)
    val_sc_loss /= len(eval_loader)
    val_od_loss /= len(eval_loader)
    val_dc_loss /= len(eval_loader)
    val_label_loss /= len(eval_loader)
    val_box_loss /= len(eval_loader)

    val_od_acc = val_od_correct / val_od_total if val_od_total > 0 else 0.0

    val_sc_cube_joint /= val_sc_cube_total
    val_sc_cube_sten  /= val_sc_cube_total
    val_sc_cube_plaq  /= val_sc_cube_total
    val_sc_vessel_sten /= val_sc_vessel_total

    val_sten_per_class_acc = [
        val_sten_correct[i] / val_sten_total[i] if val_sten_total[i] > 0 else 0.0
        for i in range(3)
    ]

    val_plaq_per_class_acc = [
        val_plaq_correct[i] / val_plaq_total[i] if val_plaq_total[i] > 0 else 0.0
        for i in range(4)
    ]
    return (val_loss, val_od_loss, val_dc_loss, val_sc_loss, val_label_loss, val_box_loss,
            val_sc_cube_joint, val_sc_cube_sten, val_sc_cube_plaq, val_sc_vessel_sten, val_od_acc, val_sten_per_class_acc, val_plaq_per_class_acc)
