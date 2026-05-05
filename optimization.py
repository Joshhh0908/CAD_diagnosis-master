import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

import functions as funcs
from functions import center_of_cube


class object_detection_loss(nn.Module):
    def __init__(self, num_classes=2, eos_coef=0.2, matcher=funcs.HungarianMatcher(), sig_weight=5):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes) #should be 7
        empty_weight[0] = self.eos_coef
        empty_weight[4:] = sig_weight #weight for significant lesions, 4-6, 1-3 non sig, 0 bg
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        empty_batch = False

        target_classes = torch.zeros(src_logits.shape[:2],
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        

        if target_classes_o.numel() != 0:
            target_classes_o = target_classes_o.to(device=src_logits.device, dtype=torch.long) 
            target_classes[idx] = target_classes_o
        else:
            empty_batch = True

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return loss_ce, empty_batch

    def loss_boxes(self, outputs, targets, indices, num_boxes, lambda_bbox=5, lambda_giou=2):

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes.to(src_boxes.device)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # loss_giou = 1 - torch.diag(funcs.generalized_box_iou(funcs.box_cxcywh_to_xyxy(src_boxes),
        #                                                     funcs.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = 1 - torch.diag(funcs.generalized_box_iou_1d(src_boxes, target_boxes))
        # NEED TO ADD HYPERPARAMETER WEIGHTS OF 5 AND 2 ACCORDING TO PAPER
        return loss_bbox.sum() / num_boxes + loss_giou.sum() / num_boxes

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        batch_idx = batch_idx.to(dtype=torch.long)
        src_idx = src_idx.to(dtype=torch.long)
        return batch_idx, src_idx

    def forward(self, outputs, targets):

        indices = self.matcher(outputs, targets)

        # print("gt counts per sample:", gt_counts)
        # print("matched counts per sample:", match_counts)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(num_boxes, dtype=torch.float, device=next(iter(outputs.values())).device)
        #this line... only working on one gpu so its unecessary? why clamp to min 1 if theres no boxes? remove the clamp
        # num_boxes = torch.clamp(num_boxes / funcs.get_world_size(), min=1).item()

        loss_labels, empty_batch = self.loss_labels(outputs, targets, indices)

        if empty_batch:
            zero = torch.tensor(0.0, device=loss_labels.device)
            total_loss = loss_labels  # no box loss
            return total_loss, loss_labels, zero

        loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)
        
        return loss_labels + loss_boxes, loss_labels, loss_boxes

#SC LOSS
class sampling_point_classification_loss(nn.Module):
    def __init__(self, num_classes=3, seq_length=32):
        super().__init__()

        self.num_classes = num_classes
        self.seq_length = seq_length
    #CE without weight for background?
    def loss_labels(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def forward(self, outputs, targets):

        logits = rearrange(outputs["pred_logits"], 'b l c -> (b l) c').to(torch.float32)
        labels = torch.cat([t["labels"] for t in targets], dim=0).to(torch.long)
        labels = labels.to(logits.device)

        return self.loss_labels(logits, labels)


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


def sc2od_targets(sc_point_data, seq_length):

    od_box_data = []
    for point_data in sc_point_data:
        tmp_data = point_data['labels']

        boxes, labels = [], []
        length = seq_length
        start, last = None, 0

        for i in range(tmp_data.shape[0]):
            if start is not None:
                if tmp_data[i] != last:
                    boxes.append([(start) / length, min((i) / length, 1.0)]) #remove +1 to start and end
                    labels.append(last) #remove the -1, 1-6 leisons, 0 bg

                    if tmp_data[i] != 0: 
                        start, last = i, tmp_data[i]
                    else:
                        start, last = None, 0
            elif tmp_data[i] != 0: 
                    start, last = i, tmp_data[i]

        if start is not None:
            boxes.append([(start) / length, 1.0])
            labels.append(last) #remove the -1, labels come in as 1-6 lesions, 0 bg
        boxes = torch.tensor(boxes, device=tmp_data.device)
        labels = torch.tensor(labels, device=tmp_data.device)  
        od_box_data.append({"labels": labels, "boxes": boxes})
    return od_box_data



class dual_task_contrastive_loss(nn.Module):
    def __init__(self, od_contrastive_loss, sc_contrastive_loss, seq_length, vessel_length, step):
        super().__init__()

        self.od_contrastive_loss = od_contrastive_loss
        self.sc_contrastive_loss = sc_contrastive_loss
        self.matcher = self.od_contrastive_loss.matcher
        self.seq_length = seq_length #num_cubes
        self.length = vessel_length
        self.step = step

    def _get_object_detection_targets(self, sc_outputs):

        ret_sc_targets = []
        for batch in sc_outputs["pred_logits"]:
            labels = torch.argmax(batch, dim=1)
            ret_sc_targets.append({"labels": labels})
        return sc2od_targets(ret_sc_targets, self.seq_length)

    def _get_sampling_point_classification_targets(self, od_outputs, od_targets):

        indices = self.matcher(od_outputs, od_targets)
        selected_indices = [item[0] for item in indices]

        ret_od_targets = []
        for batch_idx, indices in enumerate(selected_indices):

            logits = od_outputs["pred_logits"][batch_idx]
            boxes = od_outputs["pred_boxes"][batch_idx]
            selected_logits = logits[indices]
            selected_boxes = boxes[indices]
            labels = torch.argmax(selected_logits, dim=1) #removed the -1
            # labels = torch.clamp(labels, min=0)

            ret_od_targets.append({"labels": labels, "boxes": selected_boxes})

        return od2sc_targets(ret_od_targets, self.seq_length)

    def forward(self, od_outputs, sc_outputs, od_targets):

        sc_con_targets = self._get_sampling_point_classification_targets(od_outputs, od_targets)
        od_con_targets = self._get_object_detection_targets(sc_outputs)

        sc_loss_values =self.sc_contrastive_loss(sc_outputs, sc_con_targets)
        od_loss_values, loss_labels, loss_boxes = self.od_contrastive_loss(od_outputs, od_con_targets)

        return sc_loss_values + od_loss_values


class spatio_temporal_contrast_loss(nn.Module):
    def __init__(self, num_classes=2, seq_length=32, eos_coef=0.2, step=8, length=256, sig_weight=5):
        super().__init__()

        self.num_classes = num_classes
        self.seq_length = seq_length
        self.eos_coef = eos_coef

        self.od_loss = object_detection_loss(num_classes=self.num_classes, eos_coef=self.eos_coef,
                                             matcher=funcs.HungarianMatcher(), sig_weight=sig_weight)
        self.sc_loss = sampling_point_classification_loss(num_classes=self.num_classes, seq_length=self.seq_length)
        self.dc_loss = dual_task_contrastive_loss(self.od_loss, self.sc_loss, seq_length=self.seq_length, vessel_length=length,step=step)

    def forward(self, od_outputs, sc_outputs, od_targets, sc_targets, delta=0.25):

        dc = self.dc_loss(od_outputs, sc_outputs, od_targets) * delta
        od, loss_labels, loss_boxes = self.od_loss(od_outputs, od_targets)
        sc = self.sc_loss(sc_outputs, sc_targets)

        # print("dc shape:", dc.shape, "value:", dc)
        # print("od shape:", od.shape, "value:", od)
        # print("sc shape:", sc.shape, "value:", sc)

        ret_loss = dc
        ret_loss = ret_loss + od
        ret_loss = ret_loss + sc
        
        return ret_loss, sc, od, dc, loss_labels, loss_boxes
