import os
from collections import Counter

folder = "/home/joshua/CAD_diagnosis-master/data"
test  = os.path.join(folder, "test/labels")
train = os.path.join(folder, "train/labels")

def count_classes(label_dir):
    counts = Counter()
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath) as f:
            for line in f:
                val = int(line.strip())
                counts[val] += 1
    return counts

print("TRAIN:")
train_counts = count_classes(train)
total_train = sum(train_counts.values())
for cls in sorted(train_counts):
    print(f"  class {cls}: {train_counts[cls]} ({train_counts[cls]/total_train*100:.2f}%))")

print("\nTEST:")
test_counts = count_classes(test)
total_test = sum(test_counts.values())
for cls in sorted(test_counts):
    print(f"  class {cls}: {test_counts[cls]} ({test_counts[cls]/total_test*100:.2f}%))")