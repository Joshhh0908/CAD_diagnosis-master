import torch
import logging
from framework import sc_net_framework
from tqdm import tqdm
from config_2 import opt
from functions import boxes_cw_to_se, log_and_write, plot_curves, save_cms, save_model
from eval import eval_epoch
import traceback
import os
import json

def train(num_epochs=200, lr=1e-5, device='cuda:1', model_name='model_58x40x8'):
    
    base_dir = os.path.join("models", model_name)

    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    curves_dir = os.path.join(results_dir, "curves")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)

    # set up log file
    log_path = os.path.join(results_dir, "log.txt")    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()
    log.info(f"Starting training — save_path={log_path} lr={lr} device={device}")

    fw = sc_net_framework(pattern='fine_tuning', cfg=opt)
    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)
    train_loader = fw.dataLoader_train
    eval_loader  = fw.dataLoader_eval

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    config = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "lr": lr,
        "device": device,
        "optimizer": "AdamW"
    }

    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(eval_loader)}")
    print(f"Starting training for {num_epochs} epochs\n")
    
    best_od_acc = (0.0, 0)
    best_sc_acc = (0.0, 0)
    best_od_and_sc_acc = (0.0, 0)

    for epoch in range(num_epochs):
        try:
            first_batch = True

            # --- training ---
            model.train()
            model.pattern = 'training'
            model.sampling_point_framework.pattern = 'training'
            model.object_detection_framework.pattern = 'training'

            train_loss = 0.0
            train_od_loss = 0.0
            train_sc_loss = 0.0
            train_dc_loss = 0.0
            train_label_loss = 0.0
            train_box_loss = 0.0

            train_bar = tqdm(train_loader, 
                            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                            leave=False)

            for images, od_targets, sc_targets, _ in train_bar:
                images = images.to(device)

                od_targets = [{k: v.to(device) for k, v in t.items()} for t in od_targets]
                sc_targets = [{k: v.to(device) for k, v in t.items()} for t in sc_targets]

                od_outputs, sc_outputs = model(images)
                
                # first batch of each epoch (debug prints)
                if first_batch:
                    with torch.no_grad():
                        probs = torch.softmax(od_outputs['pred_logits'], dim=-1)
                        bg = probs[:,:,0].mean().item()
                        ml = probs[:,:,1:].max().item()
                    log.info(f"  [DIAG e{epoch+1:03d}] bg={bg:.3f} max_lesion={ml:.3f}")
                    first_batch = False

                # convert boxes from [centre, width] to [start, end] here, before putting into loss fn
                od_outputs_for_loss = dict(od_outputs)
                od_outputs_for_loss["pred_boxes"] = boxes_cw_to_se(od_outputs["pred_boxes"])

                loss, sc_loss, od_loss, dc_loss, loss_labels, loss_boxes = loss_fn(od_outputs_for_loss, sc_outputs, od_targets, sc_targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_sc_loss += sc_loss.item()
                train_od_loss += od_loss.item()
                train_dc_loss += dc_loss.item()
                train_label_loss += loss_labels.item()
                train_box_loss += loss_boxes.item()

                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= len(train_loader)
            train_sc_loss /= len(train_loader)
            train_od_loss /= len(train_loader)
            train_dc_loss /= len(train_loader)
            train_label_loss /= len(train_loader)
            train_box_loss /= len(train_loader)

            train_metrics = {
                "loss": train_loss,
                "od_loss": train_od_loss,
                "sc_loss": train_sc_loss,
                "dc_loss": train_dc_loss,
                "label_loss": train_label_loss,
                "box_loss": train_box_loss
            }

            # --- EVALUATION ---
            val_metrics, cms = eval_epoch(model, loss_fn, eval_loader, device, epoch, num_epochs)
            
            # ----------------SAVING STUFF----------------
            save_model(model, optimizer, checkpoints_dir, epoch, train_metrics, val_metrics)
            log_and_write(log, log_path, epoch, num_epochs, train_metrics, val_metrics)
            save_cms(cms, cm_dir, epoch)

            od_acc = val_metrics["od_vessel_sten_overall_acc"]
            sc_acc = val_metrics["sc_vessel_sten_overall_acc"]
            od_and_sc_acc = val_metrics["od_and_sc_vessel_sten_overall_acc"]
            if od_acc > best_od_acc[0]:
                best_od_acc = od_acc, epoch+1
            if sc_acc > best_sc_acc[0]:
                best_sc_acc = sc_acc, epoch+1
            if od_and_sc_acc > best_od_and_sc_acc[0]:
                best_od_and_sc_acc = od_and_sc_acc, epoch+1
            
        except Exception as e:
            log.exception(f"Crash at epoch {epoch+1}")
            traceback.print_exc()
            torch.save({
                "images": images,
                "od_targets": od_targets,
                "sc_targets": sc_targets
            }, "crash_dump.pt")
            raise e
        
    plot_curves(
        csv_path=os.path.join(results_dir, "results.csv"),
        curves_dir=curves_dir
    )

    log.info(f"best od acc: {best_od_acc[0]}, epoch: {best_od_acc[1]}")
    log.info(f"best sc acc: {best_sc_acc[0]}, epoch: {best_sc_acc[1]}")
    log.info(f"best od_and_sc acc: {best_od_and_sc_acc[0]}, epoch: {best_od_and_sc_acc[1]}")

if __name__ == '__main__':
    train(lr=3e-6, num_epochs=80, device='cuda:1', model_name='newest_weight_change')