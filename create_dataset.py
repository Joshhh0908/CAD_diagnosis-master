import os
import pandas as pd
import numpy as np
import ast

def vessel_name(vessel_label):
        vessel_label = ast.literal_eval(vessel_label)

        patient_id = vessel_label[0]
        vessel_id = vessel_label[1]
        return f"{patient_id}_{vessel_id.upper()}"

def combined_score(p,s):
    """" 
    1. NS, NC
    2. NS, M
    3. NS, C
    4. S, NC
    5. S, M
    6. S, C
    """
    if s < 3:
        return str(int(p))
    else: 
         return str(int(p + 3))
          

if __name__ == "__main__":
    csv_path = "/mnt/nas4/diskm/wangxh/ctca_no/dataset/train_data_eachfolder.csv"
    folder_path = "/mnt/nas4/diskm/wangxh/ctca_no/dataset/train_geo_02mm_clean/labels"
    out_folder = "/home/joshua/CAD_diagnosis-master/data"

    df = pd.read_csv(csv_path, header=0, index_col=0)
    filenames = df["col2"].apply(vessel_name)

    for filename in filenames:
        stenosis_txt = os.path.join(folder_path, f"{filename}_stenosis.txt")
        plaque_txt = os.path.join(folder_path, f"{filename}_plaque.txt")
        stenosis = np.loadtxt(stenosis_txt)
        plaque = np.loadtxt(plaque_txt)
        out = os.path.join(out_folder, f"{filename}.txt")
        with open(out, "w") as file:
            for p, s in zip(plaque, stenosis):
                file.write(combined_score(p, s) + "\n")