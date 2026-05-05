import torch
import argparse
from tqdm import tqdm

from framework import sc_net_framework
from config import opt as opt1
from functions import boxes_cw_to_se
from optimization import od2sc_targets

# reuse your functions
from train_32x25x8 import od_inference, sc_inference, eval_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    device = args.device

    print(f"Loading model on {device}")

    # build framework (same as training)
    fw = sc_net_framework(pattern='fine_tuning', cfg=opt1)
    model = fw.model.to(device)
    loss_fn = fw.loss_fn.to(device)
    eval_loader = fw.dataLoader_eval

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"Loaded checkpoint: {args.ckpt}")

    # run eval
    (val_loss, val_od_loss, val_dc_loss, val_sc_loss, val_label_loss, val_box_loss,
                val_sc_cube_joint, val_sc_cube_sten, val_sc_cube_plaq, val_sc_vessel_sten, val_od_acc, val_sten_per_class_acc, val_plaq_per_class_acc) = eval_epoch(model, loss_fn, eval_loader, device, epoch=0, num_epochs=1)
                
    print("\n=== EVAL RESULTS ===")
    print(f"Epoch {1:03d}/{1} | "
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
            f"val_od_acc: {val_od_acc:.4f} | ")


if __name__ == "__main__":
    main()