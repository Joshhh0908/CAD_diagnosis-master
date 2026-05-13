import torch
from tqdm import tqdm
from functions import boxes_cw_to_se


JOINT_TO_STEN   = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
JOINT_TO_PLAQUE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}


STEN_MAPPING = torch.tensor([0,1,1,1,2,2,2])
PLAQ_MAPPING = torch.tensor([0,1,2,3,1,2,3])

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

def od_inference(od_outputs, targets, eval_length, cms, score_thresh=0.05):


    pred_logits = od_outputs['pred_logits']

    for b in range(eval_length):
        gt_labels = targets[b]['labels']
        # GT vessel class
        if gt_labels.numel() == 0:
            vessel_gt = 0  # no lesions, so vessel is normal
        else:
            vessel_gt = JOINT_TO_STEN[gt_labels.max().item()]#map 1-3 to 1, 4-6 to 2
            
        vessel_pred = get_vessel_pred(pred_logits[b].cpu(), score_thresh)

        cms["od_cm"][vessel_gt, vessel_pred] += 1

def sc_inference(sc_outputs, sc_targets, cms):

    sc_logits = sc_outputs["pred_logits"]  # [B, L, 7]
    device = sc_logits.device

    # mappings
    sten_mapping = STEN_MAPPING.to(device)
    plaq_mapping = PLAQ_MAPPING.to(device)

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

    # vessel level
    gt_vessel   = sten_mapping[gt_seq.max(dim=1).values]     # [B]
    pred_vessel = sten_mapping[pred_seq.max(dim=1).values]   # [B]

    #upate confusion matrices
    idx = gt_seq.view(-1) * 7 + pred_seq.view(-1)
    cms["sc_cube_joint_cm"] += torch.bincount(idx.cpu(), minlength=49).reshape(7,7)

    idx = gt_sten.view(-1) * 3 + pred_sten.view(-1)
    cms["sc_cube_sten_cm"] += torch.bincount(idx.cpu(), minlength=9).reshape(3,3)
    
    idx = gt_plaq.view(-1) * 4 + pred_plaq.view(-1)
    cms["sc_cube_plaq_cm"] += torch.bincount(idx.cpu(), minlength=16).reshape(4,4)
    
    idx = gt_vessel * 3 + pred_vessel
    cms["sc_vessel_sten_cm"] += torch.bincount(idx.cpu(), minlength=9).reshape(3,3)

def accuracies(metrics, cms):

    metrics["od_acc"] = (cms["od_cm"].trace() / cms["od_cm"].sum()).item()
    metrics["sc_cube_joint_acc"] = (cms["sc_cube_joint_cm"].trace() / cms["sc_cube_joint_cm"].sum()).item()
    metrics["sc_cube_sten_acc"] = (cms["sc_cube_sten_cm"].trace() / cms["sc_cube_sten_cm"].sum()).item()
    metrics["sc_cube_plaq_acc"] = (cms["sc_cube_plaq_cm"].trace() / cms["sc_cube_plaq_cm"].sum()).item()
    metrics["sc_vessel_sten_acc"] = (cms["sc_vessel_sten_cm"].trace() / cms["sc_vessel_sten_cm"].sum()).item()

    # OD per-class accuracy
    cm = cms["od_cm"]
    metrics["od_sten_per_class_acc"] = (cm.diag() / cm.sum(dim=1).clamp(min=1)).tolist()

    # SC stenosis per-class accuracy
    cm = cms["sc_cube_sten_cm"]
    metrics["sc_sten_acc"] = (cm.diag() / cm.sum(dim=1).clamp(min=1)).tolist()

    # SC plaque per-class accuracy
    cm = cms["sc_cube_plaq_cm"]
    metrics["sc_plaq_acc"] = (cm.diag() / cm.sum(dim=1).clamp(min=1)).tolist()

def eval_epoch(model, loss_fn, eval_loader, device, epoch, num_epochs):

    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern = 'testing'
    model.object_detection_framework.pattern = 'testing'
    
    cms = {        
        # confusion matrices
        "od_cm": torch.zeros(3,3,dtype=torch.long),
        "sc_cube_sten_cm": torch.zeros(3,3,dtype=torch.long),
        "sc_cube_plaq_cm": torch.zeros(4,4,dtype=torch.long),
        "sc_cube_joint_cm": torch.zeros(7,7,dtype=torch.long),
        "sc_vessel_sten_cm": torch.zeros(3,3,dtype=torch.long),
        }
    
    metrics = {
        #loss
        "loss": 0.0, "od_loss": 0.0, "sc_loss": 0.0,
        "dc_loss": 0.0, "label_loss": 0.0, "box_loss": 0.0,
        
        # final accuracies
        "od_acc": 0.0,
        "sc_cube_joint_acc": 0.0,
        "sc_cube_sten_acc": 0.0,
        "sc_cube_plaq_acc": 0.0,
        "sc_vessel_sten_acc": 0.0,

        # per-class acc
        "od_sten_per_class_acc": [0.0]*3,
        "sc_sten_acc": [0.0]*3,
        "sc_plaq_acc": [0.0]*4,
    }

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
                
            batch_size = images.size(0)

            od_inference(od_outputs, od_targets, batch_size, cms)
            
            sc_inference(sc_outputs, sc_targets, cms)

            val_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    accuracies(metrics, cms)

    for key in ["loss","od_loss","sc_loss","dc_loss","label_loss","box_loss"]:
        metrics[key] /= len(eval_loader)

    return metrics, cms