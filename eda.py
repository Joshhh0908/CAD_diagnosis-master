import os
from collections import Counter
import numpy as np
from itertools import groupby

folder = "/home/joshua/CAD_diagnosis-master/data"
test  = os.path.join(folder, "test/labels")
train = os.path.join(folder, "train/labels")

def count_classes(label_dir):
    counts = Counter()
    total_lesions = 0  # Variable to keep track of total lesions
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath) as f:
            for line in f:
                val = int(line.strip())
                counts[val] += 1

        # Read the labels from the .txt file and count lesions
        labels = np.loadtxt(fpath, dtype=int)
        lesions = lesions_from_slice_labels(labels)  # Get the lesion segments
        total_lesions += len(lesions)  # Count the number of lesions

    return counts, total_lesions

def lesions_from_slice_labels(slice_labels: np.ndarray):
    """
    Convert flat slice label array [0,0,1,1,1,0,2,2,0] 
    into runs: [(start, end, label), ...], ignoring label 0 (normal).
    """
    lesions = []
    for label, group in groupby(enumerate(slice_labels, start=1), key=lambda x: x[1]):
        group = list(group)
        start = group[0][0]
        end = group[-1][0]
        if label != 0:
            lesions.append((start, end, int(label)))
    return lesions

if __name__ == "__main__":
    print("TRAIN:")
    train_counts, total_train_lesions = count_classes(train)
    total_train = sum(train_counts.values())
    for cls in sorted(train_counts):
        print(f"  class {cls}: {train_counts[cls]} ({train_counts[cls]/total_train*100:.2f}%)")
    print(f"Total number of lesions in TRAIN dataset: {total_train_lesions}")

    print("\nTEST:")
    test_counts, total_test_lesions = count_classes(test)
    total_test = sum(test_counts.values())
    for cls in sorted(test_counts):
        print(f"  class {cls}: {test_counts[cls]} ({test_counts[cls]/total_test*100:.2f}%)")
    print(f"Total number of lesions in TEST dataset: {total_test_lesions}")