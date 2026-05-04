# import torch

# def detection_targets( labels_data):
#     boxes, labels = [], []
#     length = labels_data.shape[0]
#     start, last = None, 0

#     for i in range(labels_data.shape[0]):
#         if start is not None:
#             if labels_data[i] != last:
#                 boxes.append([(start + 1) / length, min((i + 1) / length, 1.0)])
#                 labels.append(last)
#                 if labels_data[i] != 0:
#                     start, last = i, labels_data[i]
#                 else:
#                     start, last = None, 0
#         else:
#             if labels_data[i] != 0:
#                 start, last = i, labels_data[i]

#     if start is not None:
#         boxes.append([(start + 1) / length, 1.0])
#         labels.append(last)

#     labels = torch.tensor(labels, dtype=torch.int64)
#     boxes  = torch.tensor(boxes,  dtype=torch.float32)
#     return {"labels": labels, "boxes": boxes}

# tester = [0,0,0,0,1,1,1,0,0,0,3,3,3,3,3,2,2,2,2,2,0,0,0,0,0]


# def sc2od_targets(sc_point_data, seq_length):

#     od_box_data = []
#     for point_data in sc_point_data:
#         tmp_data = point_data['labels']

#         boxes, labels = [], []
#         length = seq_length
#         start, last = None, 0

#         for i in range(tmp_data.shape[0]):
#             print(i)
#             if start is not None:
#                 if tmp_data[i] != last:
#                     print(f"end at {i} with label {last.item()}")
#                     boxes.append([(start + 1) / length, min((i + 1) / length, 1.0)])
#                     print(f"box append {[(start + 1) / length, min((i + 1) / length, 1.0)]} ")
#                     # the start and i + 1 here is wrong, also wrong when ubild the data, why need to +1? 
#                     labels.append(last) #remove the -1, 1-6 leisons, 0 bg
#                     print(f"label append {last}")
#                     if tmp_data[i] != 0: 
#                         start, last = i, tmp_data[i]
#                     else:
#                         start, last = None, 0
#             elif tmp_data[i] != 0: 
#                     print(f"start at {i} with label {tmp_data[i].item()}")
#                     start, last = i, tmp_data[i]

#         if start is not None:
#             print("block reach")
#             boxes.append([(start + 1) / length, 1.0])
#             labels.append(last) #remove the -1, labels come in as 1-6 lesions, 0 bg

#         boxes = torch.tensor(boxes, device=tmp_data.device)
#         labels = torch.tensor(labels, device=tmp_data.device)  
#         od_box_data.append({"labels": labels, "boxes": boxes})
#     return od_box_data


# tester2 = [{"labels": torch.tensor([0,0,0,0,0,0,0,2,0,0,0,0,0,0])}]
# print(sc2od_targets(tester2, seq_length=14))
from functions import center_of_cube
import torch
step = 8
num_cubes = 32
centers = [step // 2 + step * i - 1 for i in range(num_cubes)]
# print(centers)
# # PROOF THAT SC2OD IS FUCKED UP
# # consider a random cube
# cube_1 = 6
# cube_2 = 24
# center_1 = centers[cube_1]
# center_2 = centers[cube_2]

# normalised_by_num_cube = [(cube_1) / 32, (cube_2+1) / 32]
# normalised_by_slices = (center_1 - 3)/256, (center_2 + 5)/256
# cube_to_slice = [int(x * 256) for x in normalised_by_num_cube]
# print(f"cube_to_slice: {cube_to_slice}")
# print(f"actual slices: {center_1}, {center_2}")
# print(f"normalised_by_num_cube: {normalised_by_num_cube}")
# print(f"normalised_by_slices: {normalised_by_slices}")

def od2sc_targets(od_box_data, seq_length):

    sc_point_data = []
    for box_data in od_box_data:
        device = box_data['boxes'].device
        point_data = torch.zeros(seq_length, dtype=torch.long, device=device)
        tmp = torch.round(box_data['boxes']*(seq_length + 1)).int()
        #change seq length to slices, then match to closes cube center, then back to cube idx
        tmp = torch.clamp(tmp, min=1, max=seq_length) - 1
        #tmp is the start and end cube indes
        # over here they do this clamp and -1 to make it 0 indexed i think
        #why just shift back one cube for what
        for k in range(tmp.shape[0]):
            point_data[tmp[k, 0]:tmp[k, 1] + 1] = box_data['labels'][k] #remove the +1, labels come in as 0-5 for lesions, 6 for bg
        sc_point_data += [{"labels": point_data}]
    return sc_point_data

import torch

def od2sc_targets(od_box_data, seq_length):
    sc_point_data = []

    for box_data in od_box_data:
        device = box_data['boxes'].device

        point_data = torch.zeros(seq_length, dtype=torch.long, device=device)

        boxes = box_data['boxes']
        labels = box_data['labels']

        # convert normalized box → physical space (same as true_cubes)
        for k in range(boxes.shape[0]):

            x0 = boxes[k, 0].item()
            x1 = boxes[k, 1].item()

            s0 = x0 * 256
            s1 = x1 * 256

            for i in range(seq_length):
                lo, hi = cube_span(i)

                # EXACT same rule as true_cubes
                if max(lo, s0) < min(hi, s1):
                    point_data[i] = labels[k]

        sc_point_data.append({"labels": point_data})

    return sc_point_data


import torch
import random

seq_length = 32
length = 256
step = 8

# ---- cube geometry ----
def cube_span(i):
    c = center_of_cube(i, step)
    return c - 3, c + 5   # [lo, hi)

# ---- OD mapping (FIXED) ----
def od_map_single(x0, x1):
    start = torch.floor(torch.tensor(x0) * seq_length).long().item()
    end   = torch.ceil(torch.tensor(x1) * seq_length).long().item() - 1  # FIX

    start = max(0, min(seq_length - 1, start))
    end   = max(0, min(seq_length - 1, end))

    return start, end

# ---- ground truth ----
def true_cubes(x0, x1):
    s0, s1 = x0 * length, x1 * length

    res = []
    for i in range(seq_length):
        lo, hi = cube_span(i)
        if max(lo, s0) < min(hi, s1):
            res.append(i)
    return res

# ---- test loop ----
bad_boxes = []

num_tests = 5000

for _ in range(num_tests):

    a, b = sorted([random.random(), random.random()])

    if b - a < 0.01:
        continue

    start, end = od_map_single(a, b)
    mapped = list(range(start, end + 1))  # explicit inclusive

    truth = true_cubes(a, b)

    if mapped != truth:
        bad_boxes.append((a, b, mapped, truth))

print(f"Total mismatches: {len(bad_boxes)}")

for i, item in enumerate(bad_boxes[:10]):
    print(f"\nMismatch {i}")
    print("box:", item[0], item[1])
    print("mapped:", item[2])
    print("truth :", item[3])