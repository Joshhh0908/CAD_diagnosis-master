import copy
import pandas as pd
import ast
import numpy as np
from itertools import groupby

def parse_triplet_intervals(
    label,
    spacing_mm: float = 0.2,
):
    """
    this is per 
    triplet_list format:
      [start_mm, end_mm, value, start_mm, end_mm, value, ...]
      or
      ["normal"]


    returns list of tuples (each tuple is a lesion): (start slice idx, end slice idx, value)
    """
    intervals = []

    if label[0] == "normal":
        return intervals

    for k in range(int(len(label)/3)):
        s_mm = label[3 * k + 0]
        e_mm = label[3 * k + 1]
        val  = label[3 * k + 2]
        try:
            s = int((float(s_mm) / spacing_mm)) + 1
            e = int((float(e_mm) / spacing_mm))
        except Exception:
            continue

        if e > s:
            intervals.append((s, e, val))    
    return intervals

def lesions_from_slice_labels(slice_labels: np.ndarray):
    """
    Convert flat slice label array [0,0,1,1,1,0,2,2,0] 
    into runs: [(start, end, label), ...], ignoring label 0 (normal).
    """
    lesions = []
    for label, group in groupby(enumerate(slice_labels, start = 1), key=lambda x: x[1]):
        group = list(group)
        start = group[0][0]
        end = group[-1][0]
        if label != 0:
            lesions.append((start, end, int(label)))
    return lesions

if __name__ == "__main__":
    csv_path = "/mnt/nas4/diskm/wangxh/ctca_no/dataset/test_data_eachfolder.csv"
    folder_path = "/mnt/nas4/diskm/wangxh/ctca_no/dataset/test_geo_02mm_clean/labels"

    df = pd.read_csv(csv_path, header=0, index_col=0)
    df = df.rename(columns={'col2': 'plaque', 'col3': 'stenosis'})

    mismatch = []

    np_plaque = df["plaque"].to_numpy()
    for i in range(len(np_plaque)):
        vessel_label = ast.literal_eval(np_plaque[i])

        patient_id = vessel_label[0]
        vessel_id = vessel_label[1]
        label = vessel_label[2:]
        csv_lesions = parse_triplet_intervals(label, spacing_mm = 0.2)
        # csv_lesions as [] if normal or list of tuples of (start_idx, end_idx, label), e.g [(21, 27, 2), (63. 70, 1)]
        # print("csv lesions:")
        # print(csv_lesions)
        txt_file_path = f"{folder_path}/{patient_id}_{vessel_id}_plaque.txt"
        txt_labels = np.loadtxt(txt_file_path, dtype=int)
        # txt labels just a np list of label per slice e.g [0,0,0,0,1,1,1,1,3,3,0,0,0]
        txt_lesions = lesions_from_slice_labels(txt_labels)
        # print("txt lesions:")
        # print(txt_lesions)
        # print()
        if csv_lesions != txt_lesions:
            mismatch.append([f"{patient_id}_{vessel_id}",csv_lesions, txt_lesions, None])

    
    for index, vessel in enumerate(mismatch):
        id, csv_lesions, txt_lesions, _ = vessel
        csv_set = set(csv_lesions)
        txt_set = set(txt_lesions)
        
        only_in_csv = csv_set - txt_set
        only_in_txt = txt_set - csv_set
        sorted_csv = sorted(list(only_in_csv), key = lambda x:x[0])
        sorted_txt = sorted(list(only_in_txt), key = lambda x:x[0])

        sorted_csv_copy = copy.deepcopy(sorted_csv)

        i=0
        while len(sorted_txt) < len(sorted_csv_copy):
            if i<len(sorted_csv_copy)-1:
                curr = sorted_csv_copy[i]
                next = sorted_csv_copy[i+1]

                if curr[2] == next[2] and curr[1] + 1 == next[0]:
                    sorted_csv_copy[i] = (curr[0], next[1], next[2])
                    del sorted_csv_copy[i+1] 
                else:
                    i += 1
            else:
                break

        if sorted_csv_copy == sorted_txt:
            mismatch[index][3] = "txt lesions can be joined to form csv lesions"

        

    misaligned = [vessel for vessel in mismatch if not vessel[3]]

    joinable = [vessel for vessel in mismatch if vessel[3]]

    
    for index, vessel in enumerate(misaligned, start=1):
        id, csv_lesions, txt_lesions, reason = vessel
        # csv_set = set(csv_lesions)
        # txt_set = set(txt_lesions)
        
        # only_in_csv = csv_set - txt_set
        # only_in_txt = txt_set - csv_set
        sorted_csv_lesions = sorted(csv_lesions, key = lambda x:x[0])
        sorted_txt_lesions = sorted(txt_lesions, key = lambda x:x[0])
        #ignore with reason
        print(f"{index}. {id}")
        print(f"CSV Lesions: {sorted_csv_lesions}")
        print(f"TXT Lesions: {sorted_txt_lesions}")
        print()
