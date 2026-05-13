import torch
import logging
import csv
import os
from framework import sc_net_framework
from tqdm import tqdm
from config import opt
from functions import boxes_cw_to_se
from eval import eval_epoch
import traceback

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

    fw = sc_net_framework(pattern='fine_tuning', cfg=opt)
    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)
    train_loader = fw.dataLoader_train
    eval_loader  = fw.dataLoader_eval

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
 
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(eval_loader)}")
    print(f"Starting training for {num_epochs} epochs\n")
    
    best_val_od_acc = 0.0

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

            # --- EVALUATION ---
            (val_loss, val_od_loss, val_dc_loss, val_sc_loss, val_label_loss, val_box_loss,
             val_sc_cube_joint, val_sc_cube_sten, val_sc_cube_plaq, val_sc_vessel_sten, val_od_acc, val_sten_per_class_acc, val_plaq_per_class_acc) = eval_epoch(model, loss_fn, eval_loader, device, epoch, num_epochs)
            
            # ----------------SAVING STUFF----------------
            epoch_path = f"{save_path}_epoch{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, epoch_path)

            log.info(f"Epoch {epoch+1:03d}/{num_epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_od_loss: {train_od_loss:.4f} | "
                    f"train_dc_loss: {train_dc_loss:.4f} | "
                    f"train_sc_loss: {train_sc_loss:.4f} | "
                    f"train_label_loss: {train_label_loss:.4f} | "
                    f"train_box_loss: {train_box_loss:.4f} | "

                    f"val: {val_loss:.4f} | "
                    f"val_od_loss: {val_od_loss:.4f} | "
                    f"val_dc_loss: {val_dc_loss:.4f} | "
                    f"val_sc_loss: {val_sc_loss:.4f} | "
                    f"val_label_loss: {val_label_loss:.4f} | "
                    f"val_box_loss: {val_box_loss:.4f} | "
                    f"val_sc_cube_joint_acc: {val_sc_cube_joint:.4f} |"
                    f"val_sc_cube_sten_acc: {val_sc_cube_sten:.4f} |" 
                    f"val_sc_cube_plaq_acc: {val_sc_cube_plaq:.4f} |"
                    f"val_sc_vessel_sten_acc: {val_sc_vessel_sten:.4f} |"
                    f"val_sten_per_class_acc: {val_sten_per_class_acc} |"
                    f"val_plaq_per_class_acc: {val_plaq_per_class_acc} |"
                    f"val_od_acc: {val_od_acc:.4f} | "
                    
                    f"{' *' if val_od_acc > best_val_od_acc else ''}")

            if val_od_acc > best_val_od_acc:
                best_val_od_acc = val_od_acc
                torch.save(model.state_dict(), f"{save_path}_best.pth")
                log.info(f"  new best val_od_acc={val_od_acc:.4f}, saved {save_path}_best.pth")

            csv_path = f"{save_path}_metrics.csv"

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
                        "val_sc_vessel_sten_acc",
                         "val_sten_per_class_acc", "val_plaq_per_class_acc", "val_od_acc"
                    ])
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss, train_od_loss, train_dc_loss, train_sc_loss,
                    train_label_loss, train_box_loss,
                    val_loss, val_od_loss, val_dc_loss, val_sc_loss,
                    val_label_loss, val_box_loss,
                    val_sc_cube_joint, val_sc_cube_sten, val_sc_cube_plaq,
                    val_sc_vessel_sten,
                     val_sten_per_class_acc, val_plaq_per_class_acc, val_od_acc
                ])

        except Exception as e:
            log.exception(f"Crash at epoch {epoch+1}")
            traceback.print_exc()
            torch.save({
                "images": images,
                "od_targets": od_targets,
                "sc_targets": sc_targets
            }, "crash_dump.pt")
            raise e
        
if __name__ == '__main__':
    train(lr=3e-6, num_epochs=80, device='cuda:0', save_path='model_weight_5_reduced_delta_fixed_sc_gt')