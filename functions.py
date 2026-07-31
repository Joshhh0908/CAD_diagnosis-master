import ast
import os
import subprocess
import time
import datetime
import pickle
from matplotlib import pyplot as plt
import numpy as np
from packaging import version
from collections import defaultdict, deque
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import Optional, List
import csv
import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from torchvision.ops.boxes import box_area
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def is_empty_tensor(tensor):
    return tensor.numel() == 0


def normalize_ct_data(data_array, hu_min=-200, hu_max=800):

    clipped_array = np.clip(data_array, hu_min, hu_max)
    normalized_array = (clipped_array - hu_min) / (hu_max - hu_min)
    return normalized_array


def _3d_cubes_selection(input_volume, cube_size, num_cubes, step, batch_size):

    b, n_l, n_h, n_w = input_volume.shape
    centers = [step // 2 + step * i - 1 for i in range(num_cubes)]
    output_cubes = torch.zeros((batch_size, len(centers), cube_size, cube_size, cube_size), device=input_volume.device)

    for i, center in enumerate(centers):
        start = center - cube_size // 2
        end = start + cube_size
        set_start, set_end = 0, cube_size
        if start < 0:
            set_start -= start
            start = 0
        if end > n_l:
            set_end = cube_size - (end - n_l)
            end = n_l
        cut_start, cut_end = int(n_h / 2 - cube_size // 2), int(n_h / 2 + cube_size // 2 + 1)
        output_cubes[:, i, set_start: set_end, :, :] = input_volume[:, start:end,
                                                                    cut_start: cut_end,
                                                                    cut_start: cut_end]
    return output_cubes


def number_parameters(Net, type_size=8):
    para = sum([np.prod(list(p.size())) for p in Net.parameters()])
    return para / 1024 * type_size / 1024


def gradient_preference(*input_tensors):
    numpy_arrays = []
    for tensor in input_tensors:
        if tensor.requires_grad:
            numpy_arrays.append(tensor.detach())
        else:
            numpy_arrays.append(tensor)
    return numpy_arrays
    
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 3, cost_giou: float = 1):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_ids = tgt_ids.to(dtype=torch.long)

        out_prob, out_bbox, tgt_ids, tgt_bbox = gradient_preference(out_prob, out_bbox, tgt_ids, tgt_bbox)

        if tgt_ids.numel() == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou_1d((out_bbox), (tgt_bbox)) #removed the dimension expansion and conversion of box formatting

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        # print("out_bbox sample:", out_bbox[:5])
        # print("tgt_bbox sample:", tgt_bbox[:5])
        # print("cost_class range:", cost_class.min().item(), cost_class.max().item())
        # print("cost_bbox range:", cost_bbox.min().item(), cost_bbox.max().item())
        # print("cost_giou range:", cost_giou.min().item(), cost_giou.max().item())
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
#generalised IOU for 1D box instead
def generalized_box_iou_1d(boxes1, boxes2):
    """
    boxes1: (N, 2) [start, end]
    boxes2: (M, 2) [start, end]
    returns: (N, M) GIoU matrix
    """
    s1, e1 = boxes1[:, 0:1], boxes1[:, 1:2]  # (N, 1)
    s2, e2 = boxes2[:, 0:1].T, boxes2[:, 1:2].T  # (1, M)

    # intersection
    inter_s = torch.max(s1, s2)   # (N, M)
    inter_e = torch.min(e1, e2)   # (N, M)
    inter = (inter_e - inter_s).clamp(min=0)

    # union
    len1 = (e1 - s1)              # (N, 1)
    len2 = (e2 - s2)              # (1, M)
    union = len1 + len2 - inter

    iou = inter / (union + 1e-6)

    # enclosing segment
    enclosing_s = torch.min(s1, s2)
    enclosing_e = torch.max(e1, e2)
    enclosing_len = (enclosing_e - enclosing_s).clamp(min=0)

    giou = iou - (enclosing_len - union) / (enclosing_len + 1e-6)
    return giou

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):

        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    def global_avg(self):
        return self.total / self.count

    def max(self):
        return max(self.deque)

    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):

    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():

    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):

    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):

    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):

    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:

    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():

    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():

    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():

    return get_rank() == 0


def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):

    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):

    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)



def masks_to_boxes(masks):

    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def center_of_cube(cube_idx, step):
    return step // 2 + step * cube_idx - 1

def boxes_cw_to_se(boxes):
    """
    Convert boxes from [center, width] to [start, end].

    Supports shape (..., 2), e.g.
    [num_boxes, 2] or [batch, num_queries, 2].
    """
    start = boxes[..., 0] - boxes[..., 1] / 2.0
    end   = boxes[..., 0] + boxes[..., 1] / 2.0
    return torch.stack((start, end), dim=-1)

def log_and_write(log, log_path, epoch, num_epochs, train_metrics, val_metrics):
    
    epoch_str = f"Epoch {epoch+1:03d}/{num_epochs} | "

    for key in train_metrics:
        epoch_str += f"  train_{key}: {train_metrics[key]:.4f} | "
    for key in val_metrics:
        val = val_metrics[key]
        if isinstance(val, list):
            val_str = str(val)  # or json.dumps(val)
            epoch_str += f"  val_{key}: {val_str} | "
        else:
            epoch_str += f"  val_{key}: {val:.4f} | "
            
    log.info(epoch_str)
    csv_path = os.path.join(os.path.dirname(log_path), "results.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ["epoch"]
            for key in train_metrics.keys():
                header.append(f"train_{key}")
            for key in val_metrics.keys():
                header.append(f"val_{key}")
            writer.writerow(header)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        values = [epoch + 1]
        for key in train_metrics.keys():
            values.append(train_metrics[key])
        for key in val_metrics.keys():
            values.append(val_metrics[key])
        writer.writerow(values)

def save_model(model, optimizer, checkpoints_dir, epoch, train_metrics, val_metrics):
    
    epoch_path = os.path.join(
        checkpoints_dir,
        f"epoch_{epoch+1:03d}.pth"
    )
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_metrics["loss"],
        'val_loss': val_metrics["loss"],
    }, epoch_path)

CMS_LABELS = {
    3: ["Normal", "Non-Significant", "Significant"],
    4: ["None", "Calcified", "Non-Calcified", "Mixed"],
    7: ["None", "NS + NC", "NS + M", "NS + C", "S + NC", "S + M", "S + C"]
}

def save_cms(cms, cm_dir, epoch):

    epoch_dir = os.path.join(cm_dir, f"epoch_{epoch+1:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    cm_path = os.path.join(epoch_dir, "cms.pth")
    torch.save(cms, cm_path)

    cms_to_plot = ["od_vessel_sten_cm", "sc_vessel_sten_cm", "od_and_sc_vessel_sten_cm"]
    for key in cms_to_plot:
        cm = cms[key]
        save_path = os.path.join(epoch_dir, f"{key}.png")
        labels = CMS_LABELS[cm.shape[0]]
        plot_cm(cm=cm, title=key, labels=labels, save_path=save_path)

def plot_cm(cm, title, labels, save_path):
    cm = cm.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # titles + labels
    ax.set_title(title, fontsize=20, pad=10)
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)

    # ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=0)

    # text annotations
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "#08306B"
            ax.text(
                j, i,
                f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=15, color=color
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_list_column(series):
    return series.apply(ast.literal_eval)


def plot_curves(csv_path, curves_dir):
    os.makedirs(curves_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # -----------------------------
    # PARSE LIST COLUMNS
    # -----------------------------
    list_columns = [
        "val_od_vessel_sten_per_class_recall",
        "val_sc_cube_sten_per_class_recall",
        "val_sc_cube_plaq_per_class_recall"
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = parse_list_column(df[col])

    # expand per-class
    od_per_class = None
    sc_sten_per_class = None
    sc_plaq_per_class = None

    if "val_od_vessel_sten_per_class_recall" in df.columns:
        od_per_class = pd.DataFrame(
            df["val_od_vessel_sten_per_class_recall"].tolist(),
            columns=["Normal", "NS", "Significant"]
        )

    if "val_sc_cube_sten_per_class_recall" in df.columns:
        sc_sten_per_class = pd.DataFrame(
            df["val_sc_cube_sten_per_class_recall"].tolist(),
            columns=["Normal", "NS", "Significant"]
        )

    if "val_sc_cube_plaq_per_class_recall" in df.columns:
        sc_plaq_per_class = pd.DataFrame(
            df["val_sc_cube_plaq_per_class_recall"].tolist(),
            columns=["Background", "Calcified", "Noncalcified", "Mixed"]
        )

    # ============================================================
    # 1. TRAIN LOSS
    # ============================================================
    plt.figure(figsize=(12, 7))

    for col in [
        "train_loss",
        "train_od_loss",
        "train_sc_loss",
        "train_dc_loss",
        "train_label_loss",
        "train_box_loss"
    ]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "train_losses.png"), dpi=300)
    plt.close()

    # ============================================================
    # 2. VALIDATION LOSS
    # ============================================================
    plt.figure(figsize=(12, 7))

    for col in [
        "val_loss",
        "val_od_loss",
        "val_sc_loss",
        "val_dc_loss",
        "val_label_loss",
        "val_box_loss"
    ]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "validation_losses.png"), dpi=300)
    plt.close()

    # ============================================================
    # 3. LOSS VS ACCURACY (PRIMARY VIEW)
    # ============================================================
    if "val_od_vessel_sten_overall_acc" in df.columns:
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.plot(df["epoch"], df["train_loss"], label="train_loss")
        ax1.plot(df["epoch"], df["val_loss"], label="val_loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        ax2 = ax1.twinx()

        for col in [
            "val_od_vessel_sten_overall_acc",
            "val_sc_vessel_sten_overall_acc",
            "val_sc_cube_sten_overall_acc",
            "val_sc_cube_plaq_overall_acc"
        ]:
            if col in df.columns:
                ax2.plot(df["epoch"], df[col], linestyle="--", label=col)

        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.title("Loss vs Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "loss_vs_accuracy.png"), dpi=300)
        plt.close()

    # ============================================================
    # 4. OD VESSEL ACCURACY (PRIMARY + PER-CLASS RECALL)
    # ============================================================
    if od_per_class is not None:
        plt.figure(figsize=(12, 7))

        if "val_od_vessel_sten_overall_acc" in df.columns:
            plt.plot(df["epoch"], df["val_od_vessel_sten_overall_acc"],
                     linewidth=3, label="Overall Accuracy")

        for col in od_per_class.columns:
            plt.plot(df["epoch"], od_per_class[col],
                     linestyle="--", label=col)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Recall")
        plt.title("OD Vessel Stenosis Performance")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "od_vessel_accuracy.png"), dpi=300)
        plt.close()

# ============================================================
# 5. SC STENOSIS ACCURACY (FIXED)
# ============================================================
        if sc_sten_per_class is not None:
            plt.figure(figsize=(12, 7))

            # -----------------------
            # OVERALL METRICS
            # -----------------------
            if "val_sc_vessel_sten_overall_acc" in df.columns:
                plt.plot(
                    df["epoch"],
                    df["val_sc_vessel_sten_overall_acc"],
                    linewidth=3,
                    label="Vessel Overall Accuracy"
                )

            if "val_sc_cube_sten_overall_acc" in df.columns:
                plt.plot(
                    df["epoch"],
                    df["val_sc_cube_sten_overall_acc"],
                    linewidth=3,
                    label="Cube Overall Accuracy"
                )

            # -----------------------
            # VESSEL-LEVEL PER CLASS
            # -----------------------
            if "val_sc_vessel_sten_per_class_recall" in df.columns:
                sc_vessel_per_class = pd.DataFrame(
                    df["val_sc_vessel_sten_per_class_recall"].tolist(),
                    columns=["Normal", "NS", "Significant"]
                )

                for col in sc_vessel_per_class.columns:
                    plt.plot(
                        df["epoch"],
                        sc_vessel_per_class[col],
                        linestyle="--",
                        label=f"Vessel {col}"
                    )

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy / Recall")
            plt.title("SC Stenosis Performance (Vessel + Cube Breakdown)")
            plt.ylim(0, 1)
            plt.legend(ncol=2)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "sc_stenosis_accuracy.png"), dpi=300)
            plt.close()

    # ============================================================
    # 6. SC PLAQUE ACCURACY
    # ============================================================
    if sc_plaq_per_class is not None:
        plt.figure(figsize=(12, 7))

        if "val_sc_cube_plaq_overall_acc" in df.columns:
            plt.plot(df["epoch"], df["val_sc_cube_plaq_overall_acc"],
                     linewidth=3, label="Overall Accuracy")

        for col in sc_plaq_per_class.columns:
            plt.plot(df["epoch"], sc_plaq_per_class[col],
                     linestyle="--", label=col)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Recall")
        plt.title("SC Plaque Performance")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "sc_plaque_accuracy.png"), dpi=300)
        plt.close()

