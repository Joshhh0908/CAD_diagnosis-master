from dataprep import CPRDataset
import numpy as np

train_ds = CPRDataset("/mnt/nas4/diskm/wangxh/CTCA_handover_26_allbranch_02to04mm_revise4_disk4/dataset/train_geo_02mm_clean/volumes/", True)

labels_file =train_ds.paths[0][3]
ret_labels = np.loadtxt(labels_file).astype(np.int32)
print(ret_labels.shape)