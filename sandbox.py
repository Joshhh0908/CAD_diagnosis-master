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

step = 8
num_cubes = 32
centers = [step // 2 + step * i - 1 for i in range(num_cubes)]
print(centers)
# consider a random cube
center_23 = centers[23]
center_24 = centers[24]

normalised_by_num_cube = [23 /32, 24/ 32]
normalised_by_slices = center_23/256, center_24/256
cube_to_slice = [int(x * 256) for x in normalised_by_num_cube]
print(f"cube_to_slice: {cube_to_slice}")
print(f"actual slices: {center_23}, {center_24}")
print(f"normalised_by_num_cube: {normalised_by_num_cube}")
print(f"normalised_by_slices: {normalised_by_slices}")
my_center_23 = center_of_cube(23, step=8)
my_center_24 = center_of_cube(24, step=8)
print(f"my_center_23: {my_center_23}")
print(f"my_center_24: {my_center_24}")
