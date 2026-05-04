import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.optim import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

from framework import sc_net_framework
import optimization as opt_fn

# ─────────────────────────────────────────────────────────────
# Label definitions
# SC: 0=bg, 1=NonSig+Calc, 2=NonSig+NonCalc, 3=NonSig+Mixed,
#          4=Sig+Calc,    5=Sig+NonCalc,    6=Sig+Mixed
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ["bg",
               "NS+NonCalc", "NS+Mix",    "NS+Calc",
               "S+NonCalc",  "S+Mix",     "S+Calc"]
CMAP = ListedColormap(
    ["black", "#aad4f5", "#5ba4cf", "#1c6fa8",
               "#f5aab4", "#e05c73", "#8b0000"]
)

# joint label → stenosis / plaque 反查表
# joint: 0=bg, 1=NS+NC, 2=NS+Mix, 3=NS+Calc, 4=S+NC, 5=S+Mix, 6=S+Calc
JOINT_TO_STEN   = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
JOINT_TO_PLAQUE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
STEN_NAMES      = ["Normal", "Non-sig", "Sig"]
PLAQUE_NAMES    = ["None", "NonCalc", "Mixed", "Calc"]


def joint_to_sten_plaque(joint_seq):
    s = np.array([JOINT_TO_STEN[int(v)]   for v in joint_seq])
    p = np.array([JOINT_TO_PLAQUE[int(v)] for v in joint_seq])
    return s, p


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def resize_seq_label_torch(label_1d, target_len):
    """Resize 1D label tensor to target_len via interval majority vote."""
    old   = label_1d.shape[0]
    out   = torch.zeros(target_len, dtype=torch.long, device=label_1d.device)
    for ni in range(target_len):
        os_ = int(ni * old / target_len)
        oe  = max(int((ni+1) * old / target_len), os_+1)
        nz  = label_1d[os_:oe]; nz = nz[nz > 0]
        if len(nz) > 0:
            vals, cnts = torch.unique(nz, return_counts=True)
            out[ni] = vals[torch.argmax(cnts)]
    return out


def move_targets_to_device(targets, device):
    return [{"boxes":  t["boxes"].to(device),
             "labels": t["labels"].to(device)} for t in targets]


def get_patient_id(name):
    return name.split("_")[0]


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def seq_to_intervals(seq):
    ivs, start, cur = [], None, 0
    for i, v in enumerate(map(int, seq)):
        if v > 0:
            if start is None: start, cur = i, v
            elif v != cur:
                ivs.append((start, i-1, cur)); start, cur = i, v
        else:
            if start is not None:
                ivs.append((start, i-1, cur)); start = None
    if start is not None:
        ivs.append((start, len(seq)-1, cur))
    return ivs


def count_detected(true_seq, pred_seq):
    ti = seq_to_intervals(true_seq); pi = seq_to_intervals(pred_seq)
    det = sum(any(not(pe<ts or ps>te) for ps,pe,_ in pi) for ts,te,_ in ti)
    fp  = sum(not any(not(pe<ts or ps>te) for ts,te,_ in ti) for ps,pe,_ in pi)
    return len(ti), det, len(ti)-det, fp


def plot_confusion_matrix(true_labels, pred_labels, class_names,
                          save_path, title="Confusion Matrix"):
    """Plot count + row-normalized confusion matrix side by side."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, ttl in [
        (axes[0], cm,      "d",    f"{title} (counts)"),
        (axes[1], cm_norm, ".2f",  f"{title} (row-normalized)")
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(labels); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(labels); ax.set_yticklabels(class_names, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = data.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        fontsize=8, color="white" if data[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_lesion_case(volume, true_seq, pred_sc_seq, save_path, case_name,
                          pred_od_seq=None, sampling_points=None):
    """
    Paper-style dual-bar visualization:
      Row 0: CPR image with red-X sampling points
      Row 1: GT stenosis bar
      Row 2: GT plaque bar
      Row 3: SC-Net stenosis bar
      Row 4: SC-Net plaque bar
      Row 5: legend

    Colors match paper Fig.3:
      Stenosis: black=No-lesion, yellow=Non-sig, orange=Sig
      Plaque:   black=None, pink=Non-calcified, purple=Mixed, blue=Calcified
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ── colour maps ───────────────────────────────────────────
    # stenosis: 0=Normal(black), 1=Non-sig(yellow), 2=Sig(orange)
    STEN_COLORS  = ["#000000", "#FFD700", "#FF8C00"]
    # plaque:   0=None(black),  1=NonCalc(pink),   2=Mixed(purple), 3=Calc(blue)
    PLAQ_COLORS  = ["#000000", "#FFB6C1", "#9370DB", "#4169E1"]

    sten_cmap = ListedColormap(STEN_COLORS)
    plaq_cmap = ListedColormap(PLAQ_COLORS)

    # ── joint → stenosis / plaque ────────────────────────────
    def to_sten(seq):
        return np.array([JOINT_TO_STEN[int(v)]  for v in seq])
    def to_plaq(seq):
        return np.array([JOINT_TO_PLAQUE[int(v)] for v in seq])

    # ── CPR strip ────────────────────────────────────────────
    strip = volume[:, volume.shape[1]//2, :] if volume.ndim == 3 else volume

    # rows: image + (GT sten, GT plaq) + (Pred sten, Pred plaq) + legend
    fig = plt.figure(figsize=(18, 5.5))
    h_ratios = [3.5, 0.55, 0.55, 0.55, 0.55, 0.65]
    gs = fig.add_gridspec(6, 1, height_ratios=h_ratios, hspace=0.12)

    # ── CPR image ────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(strip.T, cmap="gray", aspect="auto", origin="lower")
    ax_img.set_title(case_name, fontsize=9, pad=3)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    if sampling_points is not None:
        ys = np.full(len(sampling_points), strip.shape[1] // 2)
        ax_img.scatter(sampling_points, ys, marker='x', color='red',
                       s=30, linewidths=1.0, zorder=5)

    # ── 4 colour bars ─────────────────────────────────────────
    bars = [
        (to_sten(true_seq),    sten_cmap, 2, "GT Stenosis",   "GT"),
        (to_plaq(true_seq),    plaq_cmap, 3, "GT Plaque",     "GT"),
        (to_sten(pred_sc_seq), sten_cmap, 3, "Pred Stenosis", "Ours"),
        (to_plaq(pred_sc_seq), plaq_cmap, 3, "Pred Plaque",   "Ours"),
    ]
    for ri, (arr, cmap, vmax, label, _) in enumerate(bars):
        ax = fig.add_subplot(gs[ri + 1])
        ax.imshow(arr[None, :], cmap=cmap, aspect="auto",
                  vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks([0]); ax.set_yticklabels([label], fontsize=8)
        ax.set_xticks([])
        ax.tick_params(left=False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

    # x-axis on last bar
    ax.set_xlabel("Depth (slice index)", fontsize=8)
    ax.set_xticks(np.linspace(0, len(true_seq) - 1, 5).astype(int))

    # ── Legend ────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[5])
    ax_leg.axis("off")
    legend_items = [
        ("#000000", "No-lesion / None"),
        ("#FFD700", "Non-significant stenosis"),
        ("#FF8C00", "Significant stenosis"),
        ("#FFB6C1", "Non-calcified plaque"),
        ("#9370DB", "Mixed plaque"),
        ("#4169E1", "Calcified plaque"),
    ]
    handles = [plt.Line2D([0],[0], color=c, lw=8, label=l)
               for c, l in legend_items]
    # sampling point marker in legend
    handles.append(plt.Line2D([0],[0], marker='x', color='red', lw=0,
                               markersize=7, markeredgewidth=1.5, label='Sampling point'))
    ax_leg.legend(handles=handles, loc="center", ncol=4, fontsize=7.5,
                  frameon=False, handlelength=1.5, columnspacing=1.2)

    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, criterion, eval_loader, device, seq_length,
                   save_dir=None, epoch=None):
    model.eval()
    loss_sum, steps = 0.0, 0

    all_pt_pred, all_pt_true = [], []
    all_vs_pred, all_vs_true = [], []
    all_vs_sten_pred, all_vs_sten_true = [], []  # vessel level stenosis 3-class
    pat_true, pat_pred = defaultdict(list), defaultdict(list)

    vis_root = None
    if save_dir and epoch is not None:
        vis_root = os.path.join(save_dir, f"eval_epoch_{epoch}")
        os.makedirs(vis_root, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            imgs    = batch["image"].to(device)
            targets = move_targets_to_device(batch["od_target"], device)
            sc_lbl  = batch["sc_label"].to(device)     # [B, D] 0~6

            # resize D→seq_length
            sc_seq = torch.stack(
                [resize_seq_label_torch(t, seq_length) for t in sc_lbl], 0)
            sc_targets = [{"labels": t} for t in sc_seq]

            od_out, sc_out = model(imgs)
            loss = criterion(od_out, sc_out, targets, sc_targets)
            loss_sum += loss.item(); steps += 1

            pred = torch.argmax(sc_out["pred_logits"], -1)   # [B, L]
            true = sc_seq                                     # [B, L]

            for b in range(imgs.shape[0]):
                name = batch["name"][b]
                ps   = pred[b].cpu().numpy()
                ts   = true[b].cpu().numpy()

                all_pt_pred.extend(ps.tolist())
                all_pt_true.extend(ts.tolist())

                all_vs_pred.append(int(ps.max()))
                all_vs_true.append(int(ts.max()))

                # vessel level stenosis 3-class（论文图里的那种）
                # 取所有点的stenosis预测最大值作为vessel级别的stenosis
                vs_sten_p = JOINT_TO_STEN[int(ps.max())]
                vs_sten_t = JOINT_TO_STEN[int(ts.max())]
                all_vs_sten_pred.append(vs_sten_p)
                all_vs_sten_true.append(vs_sten_t)

                pid = get_patient_id(name)
                pat_true[pid].append(int(ts.max()))
                pat_pred[pid].append(int(ps.max()))

                if vis_root and step < 10:
                    # compute sampling point positions in original D=256 space
                    D = imgs[b].shape[0]
                    sp = np.linspace(0, D - 1, seq_length).astype(int)
                    # expand sc pred from seq_length back to D for visualization
                    vis_true = np.array([int(ts[int(i * seq_length / D)]) for i in range(D)])
                    vis_pred = np.array([int(ps[int(i * seq_length / D)]) for i in range(D)])
                    visualize_lesion_case(
                        volume=imgs[b].cpu().numpy(),
                        true_seq=vis_true,
                        pred_sc_seq=vis_pred,
                        save_path=os.path.join(vis_root, f"{name}.png"),
                        case_name=name,
                        sampling_points=sp,
                    )

    avg_loss = loss_sum / max(steps, 1)

    def acc(t, p): return accuracy_score(t, p) if t else 0.0

    pa_true = [max(pat_true[p]) for p in sorted(pat_true)]
    pa_pred = [max(pat_pred[p]) for p in sorted(pat_pred)]

    print("Eval SC   pred:", np.unique(all_pt_pred, return_counts=True))
    print("Eval SC   true:", np.unique(all_pt_true, return_counts=True))

    # ── stenosis / plaque 拆解 ────────────────────────────────
    all_sten_pred, all_sten_true = [], []
    all_plaq_pred, all_plaq_true = [], []
    for p, t in zip(all_pt_pred, all_pt_true):
        sp = JOINT_TO_STEN[p];  st = JOINT_TO_STEN[t]
        pp = JOINT_TO_PLAQUE[p]; pt_ = JOINT_TO_PLAQUE[t]
        all_sten_pred.append(sp); all_sten_true.append(st)
        if st > 0:   # plaque only on fg slices
            all_plaq_pred.append(pp); all_plaq_true.append(pt_)

    def acc(t, p): return accuracy_score(t, p) if t else 0.0
    pt_acc   = acc(all_pt_true,   all_pt_pred)
    vs_acc      = acc(all_vs_true,      all_vs_pred)
    vs_sten_acc = acc(all_vs_sten_true, all_vs_sten_pred)

    print("Eval VS-Sten pred:", np.unique(all_vs_sten_pred, return_counts=True))
    print("Eval VS-Sten true:", np.unique(all_vs_sten_true, return_counts=True))
    sten_acc = acc(all_sten_true, all_sten_pred)
    plaq_acc = acc(all_plaq_true, all_plaq_pred)

    pa_true = [max(pat_true[p]) for p in sorted(pat_true)]
    pa_pred = [max(pat_pred[p]) for p in sorted(pat_pred)]
    pa_acc  = acc(pa_true, pa_pred)

    print("Eval Sten pred:", np.unique(all_sten_pred, return_counts=True))
    print("Eval Plaq pred:", np.unique(all_plaq_pred, return_counts=True))

    # ── 三张混淆矩阵 ──────────────────────────────────────────
    if save_dir and epoch is not None:
        plot_confusion_matrix(all_pt_true,   all_pt_pred,   CLASS_NAMES,
            os.path.join(save_dir, f"cm_joint_epoch_{epoch}.png"),
            title=f"Joint SC  Epoch {epoch}")
        plot_confusion_matrix(all_sten_true, all_sten_pred, STEN_NAMES,
            os.path.join(save_dir, f"cm_stenosis_epoch_{epoch}.png"),
            title=f"Stenosis Point-level  Epoch {epoch}")
        plot_confusion_matrix(all_vs_sten_true, all_vs_sten_pred, STEN_NAMES,
            os.path.join(save_dir, f"cm_stenosis_vessel_epoch_{epoch}.png"),
            title=f"Stenosis Vessel-level  Epoch {epoch}")
        if all_plaq_true:
            plot_confusion_matrix(all_plaq_true, all_plaq_pred, PLAQUE_NAMES,
                os.path.join(save_dir, f"cm_plaque_epoch_{epoch}.png"),
                title=f"Plaque (fg slices)  Epoch {epoch}")

    return {
        "eval_loss":          avg_loss,
        "point_acc":          pt_acc,
        "vessel_acc":         vs_acc,
        "patient_acc":        pa_acc,
        "stenosis_acc":       sten_acc,
        "plaque_acc":         plaq_acc,
        "vessel_stenosis_acc": vs_sten_acc,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    net          = sc_net_framework(pattern="training", state_dict_root=None)
    model        = net.model.to(device)
    criterion    = net.loss_fn
    train_loader = net.dataLoader_train
    eval_loader  = net.dataLoader_eval

    optimizer    = Adam(model.parameters(), lr=3e-5)  # 降低lr，防止高权重梯度震荡

    save_dir  = "./checkpoints_paper0428"
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 100
    seq_length = 32      # must match cubeseq_length in config

    train_losses, eval_losses             = [], []
    train_pt_accs, eval_pt_accs           = [], []
    eval_vs_accs,  eval_pa_accs           = [], []
    eval_sten_accs, eval_plaq_accs        = [], []
    eval_vs_sten_accs                     = []
    best_eval_loss   = float("inf")

    for epoch in range(num_epochs):
        model.train()
        loss_sum, steps = 0.0, 0
        all_tr_pred, all_tr_true = [], []

        for step, batch in enumerate(train_loader):
            imgs    = batch["image"].to(device, dtype=torch.float32)
            targets = move_targets_to_device(batch["od_target"], device)
            sc_lbl  = batch["sc_label"].to(device)

            sc_seq     = torch.stack(
                [resize_seq_label_torch(t, seq_length) for t in sc_lbl], 0)
            sc_targets = [{"labels": t} for t in sc_seq]

            od_out, sc_out = model(imgs)
            loss = criterion(od_out, sc_out, targets, sc_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            loss_sum += loss.item(); steps += 1

            with torch.no_grad():
                pred = torch.argmax(sc_out["pred_logits"], -1)
                all_tr_pred.extend(pred.reshape(-1).cpu().numpy().tolist())
                all_tr_true.extend(sc_seq.reshape(-1).cpu().numpy().tolist())

                if step % 50 == 0:
                    print(f"[E{epoch+1}/{num_epochs}][S{step}/{len(train_loader)}] "
                          f"loss={loss.item():.4f}")
                    print(f"  pred: {np.unique(pred.cpu().numpy(), return_counts=True)}")
                    print(f"  true: {np.unique(sc_seq.cpu().numpy(), return_counts=True)}")

        avg_train = loss_sum / max(steps, 1)
        tr_pt_acc = accuracy_score(all_tr_true, all_tr_pred) if all_tr_true else 0.0

        print(f"Epoch {epoch+1} train loss      = {avg_train:.4f}")
        print(f"Epoch {epoch+1} train point acc = {tr_pt_acc:.4f}")

        metrics = evaluate_model(
            model=model, criterion=criterion, eval_loader=eval_loader,
            device=device, seq_length=seq_length,
            save_dir=save_dir, epoch=epoch+1)

        print(f"Epoch {epoch+1} eval  loss      = {metrics['eval_loss']:.4f}")
        print(f"Epoch {epoch+1} eval  point acc = {metrics['point_acc']:.4f}")
        print(f"Epoch {epoch+1} eval  vessel acc= {metrics['vessel_acc']:.4f}")
        print(f"Epoch {epoch+1} eval  patient acc={metrics['patient_acc']:.4f}")
        print(f"Epoch {epoch+1} eval  sten(pt)  ={metrics['stenosis_acc']:.4f}")
        print(f"Epoch {epoch+1} eval  sten(vs)  ={metrics['vessel_stenosis_acc']:.4f}  ← vessel level")
        print(f"Epoch {epoch+1} eval  plaque acc ={metrics['plaque_acc']:.4f}")

        # record
        train_losses.append(avg_train);         eval_losses.append(metrics['eval_loss'])
        train_pt_accs.append(tr_pt_acc);        eval_pt_accs.append(metrics['point_acc'])
        eval_vs_accs.append(metrics['vessel_acc'])
        eval_pa_accs.append(metrics['patient_acc'])
        eval_sten_accs.append(metrics['stenosis_acc'])
        eval_plaq_accs.append(metrics['plaque_acc'])
        eval_vs_sten_accs.append(metrics['vessel_stenosis_acc'])

        # checkpoint + early stopping
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pt"))
        if metrics['eval_loss'] < best_eval_loss:
            best_eval_loss = metrics['eval_loss']
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            print("  → best checkpoint saved")

        # plots
        def save_plot(path, curves):
            plt.figure()
            for lbl, data in curves: plt.plot(data, label=lbl)
            plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
            plt.savefig(path); plt.close()

        save_plot(os.path.join(save_dir, "loss_curve.png"),
                  [("train", train_losses), ("eval", eval_losses)])
        save_plot(os.path.join(save_dir, "acc_curve.png"),
                  [("train_pt", train_pt_accs), ("eval_pt", eval_pt_accs),
                   ("eval_vs", eval_vs_accs),   ("eval_pa", eval_pa_accs),
                   ("eval_sten_pt", eval_sten_accs), ("eval_sten_vs", eval_vs_sten_accs),
                   ("eval_plaq", eval_plaq_accs)])

        # CSV
        with open(os.path.join(save_dir, "training_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","eval_loss",
                        "train_pt_acc","eval_pt_acc","eval_vs_acc","eval_pa_acc",
                        "eval_sten_pt_acc","eval_sten_vs_acc","eval_plaq_acc"])
            for i in range(len(train_losses)):
                w.writerow([i+1, train_losses[i], eval_losses[i],
                             train_pt_accs[i], eval_pt_accs[i],
                             eval_vs_accs[i],  eval_pa_accs[i],
                             eval_sten_accs[i], eval_vs_sten_accs[i], eval_plaq_accs[i]])


if __name__ == "__main__":
    main()
