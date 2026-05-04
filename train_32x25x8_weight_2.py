import torch
import math
import logging
import csv
import os
from framework import sc_net_framework
from tqdm import tqdm
from config_2 import opt as opt1
from functions import boxes_cw_to_se
from optimization import od2sc_targets

JOINT_TO_STEN   = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
JOINT_TO_PLAQUE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}

def od_inference(model, eval_loader, device, score_thresh=0.05):
    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern = 'testing'
    model.object_detection_framework.pattern = 'testing'
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets, names in eval_loader:
            images = images.to(device)
            od_outputs = model(images)
            
            pred_logits = od_outputs['pred_logits']
            
            for b in range(images.shape[0]):
                gt_labels = targets[b]['labels']
                
                # GT vessel class
                if gt_labels.numel() == 0:
                    vessel_gt = 0  # no lesions, so vessel is normal
                else:
                    vessel_gt = JOINT_TO_STEN[gt_labels.max().item()]#map 1-3 to 1, 4-6 to 2
                
                
                # Pred vessel class
                pred_classes = pred_logits[b].max(dim=-1)[1]
                pred_scores  = pred_logits[b].max(dim=-1)[0]
                scores = torch.tensor(pred_scores)
                classes = torch.tensor(pred_classes)

                mask = scores >= score_thresh

                if mask.any():
                    valid_scores = scores.clone()
                    valid_scores[~mask] = -1e9   # ignore low confidence

                    idx = valid_scores.argmax().item()
                    vessel_pred = JOINT_TO_STEN[classes[idx].item()]
                else:
                    vessel_pred = 0  # fallback = Normal

                if vessel_gt == vessel_pred:
                    correct += 1
                total += 1
    
    # restore training mode 
    model.train()
    model.pattern = 'training'
    model.sampling_point_framework.pattern = 'training'
    model.object_detection_framework.pattern = 'training'
    
    return correct / total if total > 0 else 0.0

def sc_inference(model, test_loader, device, score_thresh):
    model.eval()
    model.pattern = 'testing'
    model.sampling_point_framework.pattern = 'testing'
    model.object_detection_framework.pattern = 'testing'

    # ── cube-level stats ─────────────────────────────
    cube_joint_correct = 0
    cube_joint_total   = 0

    cube_sten_correct = 0
    cube_sten_total   = 0
    per_cube_sten_correct = [0, 0, 0]
    per_cube_sten_total   = [0, 0, 0]

    cube_plaq_correct = 0
    cube_plaq_total   = 0
    per_cube_plaq_correct = [0, 0, 0, 0]
    per_cube_plaq_total   = [0, 0, 0, 0]

    # ── vessel-level stats ───────────────────────────
    vessel_joint_correct = 0
    vessel_sten_correct  = 0
    vessel_total         = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='SC eval', leave=False):
            images, targets = batch[0], batch[1]
            images = images.to(device)

            od_out, sc_out = model(images)
            sc_logits = sc_out["pred_logits"]  # [B, L, 7]

            sc_targets = od2sc_targets(targets, 32)  # [B, L]
            sc_targets = torch.stack([t['labels'] for t in sc_targets], dim=0)  # [B, L]
            
            B, L, _ = sc_logits.shape

            for b in range(B):

                # PRED
                pred_seq = sc_logits[b].argmax(dim=-1)  # [L]

                pred_joint = pred_seq
                pred_sten  = torch.tensor([JOINT_TO_STEN[int(x)] for x in pred_seq])
                pred_plaq  = torch.tensor([JOINT_TO_PLAQUE[int(x)] for x in pred_seq])

                # GT
                gt_seq = sc_targets[b]

                gt_joint = gt_seq
                gt_sten  = torch.tensor([JOINT_TO_STEN[int(x)] for x in gt_seq])
                gt_plaq  = torch.tensor([JOINT_TO_PLAQUE[int(x)] for x in gt_seq])

                # 1. CUBE-LEVEL METRICS

                # --- joint (7-class) ---
                cube_joint_correct += (pred_joint == gt_joint).sum().item()
                cube_joint_total   += L

                # --- stenosis (3-class) ---
                cube_sten_correct += (pred_sten == gt_sten).sum().item()
                cube_sten_total   += L

                for i in range(L):
                    s = int(gt_sten[i])
                    per_cube_sten_total[s] += 1
                    if pred_sten[i] == gt_sten[i]:
                        per_cube_sten_correct[s] += 1

                # --- plaque (4-class) ---
                cube_plaq_correct += (pred_plaq == gt_plaq).sum().item()
                cube_plaq_total   += L

                for i in range(L):
                    p = int(gt_plaq[i])
                    per_cube_plaq_total[p] += 1
                    if pred_plaq[i] == gt_plaq[i]:
                        per_cube_plaq_correct[p] += 1

                # 2. VESSEL-LEVEL (aggregation over cubes)

                # joint vessel
                vg = int(gt_seq.max().item())
                vp = int(pred_seq.max().item())

                vessel_total += 1
                if vg == vp:
                    vessel_joint_correct += 1

                # stenosis vessel
                vg_s = JOINT_TO_STEN[vg]
                vp_s = JOINT_TO_STEN[vp]

                if vg_s == vp_s:
                    vessel_sten_correct += 1


    # FINAL METRICS

    cube_joint_acc = cube_joint_correct / cube_joint_total

    cube_sten_acc = cube_sten_correct / cube_sten_total
    cube_plaq_acc = cube_plaq_correct / cube_plaq_total

    vessel_joint_acc = vessel_joint_correct / vessel_total
    vessel_sten_acc  = vessel_sten_correct / vessel_total

    return {
        "cube_joint_acc": cube_joint_acc,
        "cube_sten_acc": cube_sten_acc,
        "cube_plaq_acc": cube_plaq_acc,

        "cube_sten_per_class": per_cube_sten_correct,
        "cube_plaq_per_class": per_cube_plaq_correct,

        "vessel_joint_acc": vessel_joint_acc,
        "vessel_sten_acc": vessel_sten_acc,
    }

def train(num_epochs=200, lr=1e-5, device='cuda:1', save_path='model_58x40x8'):
    # set up log file
    log_path = f"{save_path}_train.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()
    log.info(f"Starting training — save_path={save_path} lr={lr} device={device}")

    fw = sc_net_framework(pattern='fine_tuning', cfg=opt1)
    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)
    train_loader = fw.dataLoader_train
    eval_loader  = fw.dataLoader_eval

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


    
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(eval_loader)}")
    print(f"Starting training for {num_epochs} epochs\n")
    

    for epoch in range(num_epochs):
        first_batch = True

        # --- training ---
        model.train()
        model.pattern = 'training'
        model.sampling_point_framework.pattern = 'training'
        model.object_detection_framework.pattern = 'training'

        train_loss = 0.0
        od_loss = 0.0
        sc_loss = 0.0
        dc_loss = 0.0
        loss_labels = 0.0
        loss_boxes = 0.0

        train_bar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                         leave=False)
        for images, targets, names in train_bar:
            images = images.to(device)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            od_outputs, sc_outputs = model(images)
            # convert boxes from [centre, width] to [start, end] here, before putting into loss fn
            
            # first batch of each epoch
            if first_batch:
                with torch.no_grad():
                    probs = torch.softmax(od_outputs['pred_logits'], dim=-1)
                    bg = probs[:,:,-1].mean().item()
                    ml = probs[:,:,:-1].max().item()
                log.info(f"  [DIAG e{epoch+1:03d}] bg={bg:.3f} max_lesion={ml:.3f}")
                first_batch = False

            od_outputs_for_loss = dict(od_outputs)
            od_outputs_for_loss["pred_boxes"] = boxes_cw_to_se(od_outputs["pred_boxes"])

            # box checking debug
            # boxes = od_outputs["pred_boxes"].reshape(-1, 2)

            loss, sc_loss, od_loss, dc_loss, loss_labels, loss_boxes = loss_fn(od_outputs_for_loss, sc_outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            sc_loss += sc_loss.item()
            od_loss += od_loss.item()
            dc_loss += dc_loss.item()
            loss_labels += loss_labels.item()
            loss_boxes += loss_boxes.item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # --- validation ---
        model.train()
        model.pattern = 'training'
        model.sampling_point_framework.pattern = 'training'
        model.object_detection_framework.pattern = 'training'

        val_loss = 0.0
        val_sc_loss = 0.0
        val_od_loss = 0.0
        val_dc_loss = 0.0
        val_box_loss = 0.0
        val_label_loss = 0.0

        val_bar = tqdm(eval_loader,
                       desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ",
                       leave=False)
        with torch.no_grad():
            for images, targets, _ in val_bar:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                od_outputs, sc_outputs = model(images)
                od_outputs_for_loss = dict(od_outputs)
                od_outputs_for_loss["pred_boxes"] = boxes_cw_to_se(od_outputs["pred_boxes"])

                # box checking debug
                # boxes = od_outputs["pred_boxes"].reshape(-1, 2)
                loss, sc_loss, od_loss, dc_loss, loss_labels, loss_boxes = loss_fn(od_outputs_for_loss, sc_outputs, targets)

                val_loss += loss.item()
                val_sc_loss += sc_loss.item()
                val_od_loss += od_loss.item()
                val_dc_loss += dc_loss.item()
                val_label_loss += loss_labels.item()
                val_box_loss += loss_boxes.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(eval_loader)

        
        # save every epoch
        epoch_path = f"{save_path}_epoch{epoch+1:03d}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, epoch_path)

        # after val loss computation each epoch:
        val_od_acc = od_inference(model, eval_loader, device)
        sc_metrics = sc_inference(model, eval_loader, device)
        val_sc_cube_joint = sc_metrics["cube_joint_acc"]
        val_sc_cube_sten  = sc_metrics["cube_sten_acc"]
        val_sc_cube_plaq  = sc_metrics["cube_plaq_acc"]

        val_sc_vessel_joint = sc_metrics["vessel_joint_acc"]
        val_sc_vessel_sten  = sc_metrics["vessel_sten_acc"]

        log.info(f"Epoch {epoch+1:03d}/{num_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_od_loss: {od_loss.item():.4f} | "
                f"train_dc_loss: {dc_loss.item():.4f} | "
                f"train_sc_loss: {sc_loss.item():.4f} | "
                f"train_label_loss: {loss_labels.item():.4f} | "
                f"train_box_loss: {loss_boxes.item():.4f} | "

                f"val: {val_loss:.4f} | "
                f"val_od_loss: {val_od_loss:.4f} | "
                f"val_dc_loss: {val_dc_loss:.4f} | "
                f"val_sc_loss: {val_sc_loss:.4f} | "
                f"val_label_loss: {val_label_loss:.4f} | "
                f"val_box_loss: {val_box_loss:.4f} | "
                f"val_sc_cube_joint_acc: {val_sc_cube_joint:.4f} |"
                f"val_sc_cube_sten_acc: {val_sc_cube_sten:.4f} "| 
                f"val_sc_cube_plaq_acc: {val_sc_cube_plaq:.4f} |"
                f"val_sc_vessel_joint_acc: {val_sc_vessel_joint:.4f} |"
                f"val_sc_vessel_sten_acc: {val_sc_vessel_sten:.4f} |" 

                f"val_od_acc: {val_od_acc:.4f} | "
                
                f"{' *' if val_od_acc > best_val_od_acc else ''}")

        if val_od_acc > best_val_od_acc:
            best_val_od_acc = val_od_acc
            torch.save(model.state_dict(), f"{save_path}_best.pth")
            log.info(f"  new best val_od_acc={val_od_acc:.4f}, saved {save_path}_best.pth")

        csv_path = f"{save_path}_metrics.csv"

        # create file + header if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss", "train_od_loss", "train_dc_loss", "train_sc_loss",
                    "train_label_loss", "train_box_loss",
                    "val_loss", "val_od_loss", "val_dc_loss", "val_sc_loss",
                    "val_label_loss", "val_box_loss",
                    "val_sc_cube_joint_acc", "val_sc_cube_sten_acc", "val_sc_cube_plaq_acc",
                    "val_sc_vessel_joint_acc", "val_sc_vessel_sten_acc",
                    "val_od_acc"
                ])
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss, od_loss.item(), dc_loss.item(), sc_loss.item(),
                loss_labels.item(), loss_boxes.item(),
                val_loss, val_od_loss, val_dc_loss, val_sc_loss,
                val_label_loss, val_box_loss,
                val_sc_cube_joint, val_sc_cube_sten, val_sc_cube_plaq,
                val_sc_vessel_joint, val_sc_vessel_sten,
                val_od_acc
            ])

if __name__ == '__main__':
    train(lr=3e-6, num_epochs=80, device='cuda:1', save_path='model_32x25x8_weight_2')