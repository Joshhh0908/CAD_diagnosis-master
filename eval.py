import numpy as np
import torch
from tqdm import tqdm
from functions import boxes_cw_to_se


JOINT_TO_STEN   = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
JOINT_TO_PLAQUE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}


STEN_MAPPING = torch.tensor([0,1,1,1,2,2,2])
PLAQ_MAPPING = torch.tensor([0,1,2,3,1,2,3])

BG_IDX = 0

def get_vessel_pred(pred_logits_b, score_thresh):

    pred_classes = pred_logits_b.argmax(dim=-1)
    pred_scores = pred_logits_b.max(dim=-1).values

    order = torch.argsort(pred_scores, descending=True)

    vessel_pred = 0
    best_score = 0.0

    for idx in order:

        cls = pred_classes[idx].item()
        score = pred_scores[idx].item()

        if score < score_thresh:
            continue

        vessel_pred = JOINT_TO_STEN[cls]
        best_score = score
        break

    return vessel_pred, best_score

def od_inference(od_outputs, targets, cms, score_thresh=0.05):

    pred_logits = od_outputs['pred_logits']

    vessel_preds = []
    vessel_preds_scores = []
    vessel_gts = []
    B = pred_logits.size(0)
    for b in range(B):
        gt_labels = targets[b]['labels']
        # GT vessel class
        if gt_labels.numel() == 0:
            vessel_gt = 0  # no lesions, so vessel is normal
        else:
            vessel_gt = JOINT_TO_STEN[gt_labels.max().item()]#map 1-3 to 1, 4-6 to 2
        
        vessel_gts.append(vessel_gt)
        vessel_pred, vessel_pred_score = get_vessel_pred(pred_logits[b], score_thresh)
        vessel_preds.append(vessel_pred)
        vessel_preds_scores.append(vessel_pred_score)
    vessel_preds = torch.tensor(vessel_preds, device=pred_logits.device)
    vessel_preds_scores = torch.tensor(vessel_preds_scores, device=pred_logits.device)
    vessel_gts = torch.tensor(vessel_gts, device=pred_logits.device)
    update_cm(cms["od_vessel_sten_cm"], vessel_gts, vessel_preds, 3)

    return vessel_preds, vessel_preds_scores

def sc_inference(sc_outputs, sc_targets, sten_mapping, plaq_mapping, cms):

    sc_logits = sc_outputs["pred_logits"]  # [B, L, 7]
    device = sc_logits.device


    # targets
    gt_seq = torch.stack([t['labels'].to(device) for t in sc_targets], dim=0)

    # predictions
    pred_seq = sc_logits.argmax(dim=-1)  # [B, L]
    pred_seq_scores = sc_logits.max(dim=-1).values    # [B, L]
    
    # mapped classes
    pred_sten = sten_mapping[pred_seq]
    gt_sten   = sten_mapping[gt_seq]

    pred_plaq = plaq_mapping[pred_seq]
    gt_plaq   = plaq_mapping[gt_seq]

    # vessel level
    gt_vessel   = sten_mapping[gt_seq.max(dim=1).values]     # [B]
    pred_vessel = sten_mapping[pred_seq.max(dim=1).values]   # [B]
    
    # highest score of highest severity class per vessel
    mask = (pred_sten == pred_vessel.unsqueeze(1))
    masked_scores = pred_seq_scores.masked_fill(~mask, -1)
    best_cube_idx = masked_scores.argmax(dim=1)
    pred_vessel_scores = pred_seq_scores[torch.arange(pred_seq.size(0), device=device), best_cube_idx]
    
    
    #upate confusion matrices

    update_cm(cms["sc_cube_joint_cm"], gt_seq, pred_seq, 7)
    update_cm(cms["sc_cube_sten_cm"], gt_sten, pred_sten, 3)
    update_cm(cms["sc_cube_plaq_cm"], gt_plaq, pred_plaq, 4)
    update_cm(cms["sc_vessel_sten_cm"], gt_vessel, pred_vessel, 3)


    return pred_vessel,pred_vessel_scores, gt_vessel

def joint_inference(sc_pred, sc_scores, od_pred, od_scores, vessel_gt, cms):

    joint_pred = torch.where(sc_pred > od_pred, sc_pred, od_pred)

    update_cm(cms["od_and_sc_vessel_sten_cm"], vessel_gt, joint_pred, 3)

def compute_metrics(metrics, cms):
    for key, value in cms.items():
        cm = value
        prefix = key.replace("_cm", "")

        if isinstance(cm, torch.Tensor):
            cm = cm.detach().cpu().numpy()

        total_samples = cm.sum()
        total_correct = np.trace(cm)

        # overall accuracy
        overall_acc = total_correct / max(total_samples, 1)

        num_classes = cm.shape[0]
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []

        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP

            precision = TP / max(TP + FP, 1)
            recall = TP / max(TP + FN, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_f1.append(f1)

        # macro averages
        macro_precision = np.mean(per_class_precision) if per_class_precision else 0
        macro_recall = np.mean(per_class_recall) if per_class_recall else 0
        macro_f1 = np.mean(per_class_f1) if per_class_f1 else 0

        # save into metrics dict
        metrics[f"{prefix}_overall_acc"] = overall_acc

        metrics[f"{prefix}_per_class_recall"] = per_class_recall

        metrics[f"{prefix}_macro_precision"] = macro_precision
        metrics[f"{prefix}_macro_recall"] = macro_recall
        metrics[f"{prefix}_macro_f1"] = macro_f1

def update_cm(cm, gt, pred, num_classes):
    gt = gt.view(-1)
    pred = pred.view(-1)

    idx = gt * num_classes + pred
    binc = torch.bincount(idx, minlength=num_classes*num_classes)

    cm += binc.reshape(num_classes, num_classes)

def eval_epoch(model, loss_fn, eval_loader, device, epoch, num_epochs):

    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern = 'testing'
    model.object_detection_framework.pattern = 'testing'
    
    cms = {        
        # confusion matrices
        "od_vessel_sten_cm": torch.zeros(3,3,dtype=torch.long, device=device),
        "sc_cube_sten_cm": torch.zeros(3,3,dtype=torch.long, device=device),
        "sc_cube_plaq_cm": torch.zeros(4,4,dtype=torch.long, device=device),
        "sc_cube_joint_cm": torch.zeros(7,7,dtype=torch.long, device=device),
        "sc_vessel_sten_cm": torch.zeros(3,3,dtype=torch.long, device=device),
        "od_and_sc_vessel_sten_cm": torch.zeros(3,3,dtype=torch.long, device=device),
        }

    #dont actually need to init everything to 0... just init empty dict and write in later
    #this initialisation is more for clarity. (for now)
    # but initialising this using a loop with the prefixes would be cleaner
    metrics = {
    #loss
    "loss": 0.0, "od_loss": 0.0, "sc_loss": 0.0,
    "dc_loss": 0.0, "label_loss": 0.0, "box_loss": 0.0,
    
    # overall accuracies
    "od_vessel_sten_overall_acc": 0.0,
    "sc_cube_sten_overall_acc": 0.0,
    "sc_cube_plaq_overall_acc": 0.0,
    "sc_cube_joint_overall_acc": 0.0,
    "sc_vessel_sten_overall_acc": 0.0,
    "od_and_sc_vessel_sten_overall_acc": 0.0,
    
    # macro_precision
    "od_vessel_sten_macro_precision": 0.0,
    "sc_cube_sten_macro_precision": 0.0,
    "sc_cube_plaq_macro_precision": 0.0,
    "sc_vessel_sten_macro_precision": 0,
    "od_and_sc_vessel_sten_macro_precision": 0,  
    
    # macro_recall
    "od_vessel_sten_macro_recall": 0.0,
    "sc_cube_sten_macro_recall": 0.0,
    "sc_cube_plaq_macro_recall": 0.0,
    "sc_vessel_sten_macro_recall": 0,
    "od_and_sc_vessel_sten_macro_recall": 0,
    
    # macro_f1
    "od_vessel_sten_macro_f1": 0.0,
    "sc_cube_sten_macro_f1": 0.0,
    "sc_cube_plaq_macro_f1": 0.0,
    "sc_vessel_sten_macro_f1": 0,
    "od_and_sc_vessel_sten_macro_f1": 0,

    # per_class_recall
    "od_vessel_sten_per_class_recall": [0.0]*3,
    "sc_cube_sten_per_class_recall": [0.0]*3,
    "sc_cube_plaq_per_class_recall": [0.0]*4,
    "sc_vessel_sten_per_class_recall": [0.0]*3,
    "od_and_sc_vessel_sten_per_class_recall": [0.0]*3,
    }

    # mappings
    sten_mapping = STEN_MAPPING.to(device)
    plaq_mapping = PLAQ_MAPPING.to(device)

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

            loss, sc_loss, od_loss, dc_loss, loss_labels, loss_boxes = loss_fn(od_outputs_for_loss, sc_outputs, od_targets, sc_targets)

            paired_loss = zip(["loss","od_loss","sc_loss","dc_loss","label_loss","box_loss"],
                [loss, od_loss, sc_loss, dc_loss, loss_labels, loss_boxes])
            for key, val in paired_loss:
                metrics[key] += val.item()
                

            od_pred, od_scores = od_inference(od_outputs, od_targets, cms)
            
            sc_pred, sc_scores, gt_vessel = sc_inference(sc_outputs, sc_targets, sten_mapping, plaq_mapping, cms)

            joint_inference(sc_pred, sc_scores, od_pred, od_scores, gt_vessel, cms)

            val_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    compute_metrics(metrics, cms)

    for key in ["loss","od_loss","sc_loss","dc_loss","label_loss","box_loss"]:
        metrics[key] /= len(eval_loader)

    return metrics, cms