import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import shutil

def parse_labels(label_str):
    try:
        if isinstance(label_str, str):
            label_str = label_str.strip().strip('"').strip("'")
        label_list = ast.literal_eval(label_str)
        if len(label_list) < 2:
            return []
        lesions = []
        for i in range(2, len(label_list) - 2, 3):
            lesions.append((int(label_list[i]/0.2), int(label_list[i + 1]/0.2), int(label_list[i + 2])))
        return lesions
    except:
        return []

def make_labels(plaque, stenosis, orig_len):

    labels_p = np.zeros(orig_len, dtype=np.int32)
    labels_s = np.zeros(orig_len, dtype=np.int32)

    for start, end, ptype in plaque:
        s, e = int(round(start)), int(round(end))
        labels_p[max(0, s):min(e, orig_len)] = ptype

    for start, end, slevel in stenosis:
        s, e = int(round(start)), int(round(end))
        labels_s[max(0, s):min(e, orig_len)] = slevel

    return labels_p, labels_s

def preprocess_vessel_mask(vessel_mask_arr):
    """
    Binarize vessel mask and keep only the largest connected component.
    Uses SimpleITK ConnectedComponent + RelabelComponent (sorted by size).
    Returns a binary uint8 numpy array of the same shape.
    """
    # Step 1: binarize (any nonzero → 1)
    binary = (vessel_mask_arr > 0).astype(np.uint8)

    # Step 2: connected component labeling
    binary_sitk = sitk.GetImageFromArray(binary)
    cc = sitk.ConnectedComponent(binary_sitk)

    # Step 3: relabel by descending size (largest component = label 1)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    relabeled_arr = sitk.GetArrayFromImage(relabeled)

    # Step 4: keep only the largest component
    largest = (relabeled_arr == 1).astype(np.uint8)
    return largest


def copy_files_safe(src, dst):
    """
    Safely copy file with directory creation
    """
    try:
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            return True
        else:
            print(f"Warning: Source file not found: {src}")
            return False
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False

# Paths
# each case
def process_case(each_tr, p1, p2):
    try:
        patient_id = each_tr[0]
        image_path = each_tr[1]
        # lumen_mask_path = each_tr[4]
        # vessel_mask_path = each_tr[5]

        #
        img = sitk.ReadImage(image_path)
        volume = sitk.GetArrayFromImage(img)
        # volume = volume[::2, ::2, ::2]
        # lumen_mask = sitk.ReadImage(lumen_mask_path)
        # lumen_mask = sitk.GetArrayFromImage(lumen_mask)
        # binarize + largest CC
        #volume_mask = volume_mask[::2, ::2, ::2]
        plaque = parse_labels(each_tr[2])
        stenosis = parse_labels(each_tr[3])
        labels_p, labels_s = make_labels(plaque, stenosis, volume.shape[0])

        #
        vessel = ast.literal_eval(each_tr[2])[1]
        id = each_tr[0]

        # if os.path.exists(vessel_mask_path):
        #     vessel_mask = sitk.ReadImage(vessel_mask_path)
        #     vessel_mask = sitk.GetArrayFromImage(vessel_mask)
        #     vessel_mask = preprocess_vessel_mask(vessel_mask)
        #     if vessel_mask.shape[0] != len(labels_s):
        #         zoom_factor = len(labels_s) / vessel_mask.shape[0]
        #         vessel_mask = zoom(vessel_mask, (zoom_factor, 1, 1), order=0)  # order=0
        #         vessel_mask = (vessel_mask > 0.5).astype(vessel_mask.dtype)
        #
        #     label = vessel_mask * labels_s.reshape(len(labels_s), 1, 1)
        #     sitk.WriteImage(sitk.GetImageFromArray(vessel_mask), p4 + id + "_" + vessel + '.nii.gz')
        #     if sum(sum(sum(vessel_mask))) == 0:
        #         a = 0
        #         print(each_tr[1])
        # else:
        #     label = lumen_mask * labels_s.reshape(len(labels_s), 1, 1)

        #label = sitk.GetImageFromArray(label)

        sitk.WriteImage(sitk.GetImageFromArray(volume), p1+id+"_"+vessel+'.nii.gz')
        #sitk.WriteImage(sitk.GetImageFromArray(lumen_mask), p3 + id + "_" + vessel + '.nii.gz')

        #sitk.WriteImage(label, p5 + id + "_" + vessel + '.nii.gz')
        # nib.save(nib.Nifti1Image(volume_update.astype(np.float32).transpose(2, 1, 0), np.eye(4)), p1+id+"_"+vessel+'.nii')
        np.savetxt(p2+id+"_"+vessel+'_stenosis.txt', labels_s, fmt='%d')
        np.savetxt(p2 + id + "_" + vessel + '_plaque.txt', labels_p, fmt='%d')
        return True
    except Exception as e:
        print(each_tr[1])
        return False


if __name__ == '__main__':

    path = "/mnt/nas4/diskm/wangxh/ctca_no/dataset/test_data_eachfolder.csv"
    output_dir = ""
    #
    p1 = output_dir + "volumes/"
    p2 = output_dir + "labels/"
    # p3 = output_dir + "lumen_masks/"
    # p4 = output_dir + "vessel_masks/"
    # p5 = output_dir + "masks_label/"
    for p in [p1, p2]:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)
    #
    df = pd.read_csv(path, index_col=0)
    tr = df.values.tolist()
    success = 0
    for each_tr in tr:
        # each_tr = ["AR-NUH193",	"/mnt/nas8/Jupiter/APOLLO/revised cases and vessel masks/3rd review/24th/AR-NUH193/2016-11-10/CTCA/CPR_0.2_all/CPR4_vessel_6_rca_Centerline.mhd","['AR-NUH193', 'RCA', 10.5, 31.5, 2, 112.0, 118.0, 3]","['AR-NUH193', 'RCA', 10.5, 31.5, 1, 112.0, 118.0, 1]",
        #           "/mnt/nas8/Jupiter/APOLLO/revised cases and vessel masks/3rd review/24th/AR-NUH193/2016-11-10/CTCA/CPR_0.2_all/CPR4_vessel_6_rca_Centerline-mask.mhd",
        #            "/mnt/nas8/Jupiter/APOLLO/revised cases and vessel masks/3rd review/24th/AR-NUH193/2016-11-10/CTCA/CPR_SingleVessel_0.2/CPR4_vessel_6_rca_Centerline-Vessel_maskSingle.mhd"]
        if process_case(each_tr, p1, p2):
            success += 1

    print(f"\n: {success}/{len(df)}")
    print(f"\n: {success}/{len(df)}")