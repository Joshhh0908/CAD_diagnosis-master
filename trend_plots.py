
import pandas as pd
import matplotlib.pyplot as plt
import ast

# ---------------- LOAD CSV ----------------
csv_path = "model_weight_10_new_eval_format_metrics.csv"
df = pd.read_csv(csv_path)

# ---------------- HELPER ----------------
def parse_list_column(col):
    return col.apply(ast.literal_eval)

# Parse list columns
list_columns = [
    "val_od_sten_per_class_acc",
    "val_sc_sten_acc",
    "val_sc_plaq_acc"
]

for col in list_columns:
    df[col] = parse_list_column(df[col])

# ---------------- EXTRACT PER-CLASS METRICS ----------------
# OD vessel stenosis per-class
od_per_class = pd.DataFrame(
    df["val_od_sten_per_class_acc"].tolist(),
    columns=["Normal", "NS", "Significant"]
)

# SC vessel/cube stenosis per-class
sc_sten_per_class = pd.DataFrame(
    df["val_sc_sten_acc"].tolist(),
    columns=["Normal", "NS", "Significant"]
)

# SC plaque per-class
sc_plaq_per_class = pd.DataFrame(
    df["val_sc_plaq_acc"].tolist(),
    columns=["Background", "Calcified", "Noncalcified", "Mixed"]
)

# ============================================================
# 1. TRAIN LOSSES
# ============================================================
plt.figure(figsize=(12,7))

train_loss_columns = [
    "train_loss",
    "train_od_loss",
    "train_sc_loss",
    "train_dc_loss",
    "train_label_loss",
    "train_box_loss"
]

for col in train_loss_columns:
    plt.plot(df["epoch"], df[col], label=col)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_losses.png", dpi=300)
plt.close()

# ============================================================
# 2. VALIDATION LOSSES
# ============================================================
plt.figure(figsize=(12,7))

val_loss_columns = [
    "val_loss",
    "val_od_loss",
    "val_sc_loss",
    "val_dc_loss",
    "val_label_loss",
    "val_box_loss"
]

for col in val_loss_columns:
    plt.plot(df["epoch"], df[col], label=col)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_losses.png", dpi=300)
plt.close()

# ============================================================
# 3. MAIN LOSS VS MAIN ACCURACIES
# ============================================================
fig, ax1 = plt.subplots(figsize=(12,7))

# Loss axis
ax1.plot(df["epoch"], df["train_loss"], label="train_loss")
ax1.plot(df["epoch"], df["val_loss"], label="val_loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Accuracy axis
ax2 = ax1.twinx()

accuracy_columns = [
    "val_od_acc",
    "val_sc_vessel_sten_acc",
    "val_sc_cube_sten_acc"
]

for col in accuracy_columns:
    ax2.plot(df["epoch"], df[col], linestyle="--", label=col)

ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

plt.title("Loss vs Accuracy")
plt.tight_layout()
plt.savefig("loss_vs_accuracy.png", dpi=300)
plt.close()

# ============================================================
# 3. OD VESSEL ACCURACY
# ============================================================
plt.figure(figsize=(12,7))

plt.plot(df["epoch"], df["val_od_acc"], linewidth=3, label="Overall")

for col in od_per_class.columns:
    plt.plot(df["epoch"], od_per_class[col], linestyle="--", label=col)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("OD Vessel Stenosis Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("od_vessel_accuracy.png", dpi=300)
plt.close()

# ============================================================
# 4. SC VESSEL / CUBE STENOSIS ACCURACY
# ============================================================
plt.figure(figsize=(12,7))

plt.plot(
    df["epoch"],
    df["val_sc_vessel_sten_acc"],
    linewidth=3,
    label="SC Vessel Overall"
)

plt.plot(
    df["epoch"],
    df["val_sc_cube_sten_acc"],
    linewidth=3,
    label="SC Cube Overall"
)

for col in sc_sten_per_class.columns:
    plt.plot(df["epoch"], sc_sten_per_class[col], linestyle="--", label=f"SC {col}")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SC Stenosis Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sc_stenosis_accuracy.png", dpi=300)
plt.close()

# ============================================================
# 5. SC PLAQUE ACCURACY
# ============================================================
plt.figure(figsize=(12,7))

plt.plot(
    df["epoch"],
    df["val_sc_cube_plaq_acc"],
    linewidth=3,
    label="Overall"
)

for col in sc_plaq_per_class.columns:
    plt.plot(df["epoch"], sc_plaq_per_class[col], linestyle="--", label=col)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SC Plaque Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sc_plaque_accuracy.png", dpi=300)
plt.close()
