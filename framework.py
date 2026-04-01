import os
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader

from architecture import spatio_temporal_semantic_learning
from config_1 import opt
import functions as funcs
import optimization as opt_fn
import augmentation as aug
from config_1 import opt as default_opt


class sc_net_framework:

    def __init__(self, pattern='pre_training', state_dict_root=None, cfg=None):
        self.opt = cfg if cfg is not None else default_opt  # use passed config or default

        if pattern == "pre_training":
            self.model_pattern = "training"
            self.model_num_classes = self.opt.net_params["num_classes"][0]
        elif pattern == "fine_tuning":
            self.model_pattern = "training"
            self.model_num_classes = self.opt.net_params["num_classes"][1]
        else:
            self.model_pattern = "testing"
            self.model_num_classes = self.opt.net_params["num_classes"][1]

        self.model = self.get_model()
        self.state_dict_root = state_dict_root
        if self.state_dict_root is not None:
            self.pre_training_load()

        if pattern != 'inference':

            self.data_root = self.opt.data_params["dataset_root"]
            self.train_ratio = self.opt.data_params["train_ratio"]
            self.input_shape = self.opt.net_params["input_shape"]
            self.window_lw = self.opt.data_params["window_lw"]
            self.batch_size = self.opt.data_params["batch_size"]

            self.loss_fn = self.get_loss_fn()
            self.dataLoader_train, self.dataLoader_eval = self.get_dataloader()

    def get_model(self):
        return spatio_temporal_semantic_learning(
            num_classes=self.model_num_classes,
            pattern=self.model_pattern,
            ret_map=self.opt.net_params["ret_map"],
            in_channels=self.opt.net_params["in_channels"],
            _3d_cube_selection=self.opt.sc_params["_3d_cube_selection"],
            temporal_conv_levels=self.opt.sc_params["temporal_conv_levels"],
            temporal_conv_maps=self.opt.sc_params["temporal_conv_maps"],
            temporal_feature_channels=self.opt.sc_params["temporal_feature_channels"],
            temporal_embedding_dim=self.opt.sc_params["temporal_embedding_dim"],
            temporal_transfromer_param=self.opt.sc_params["temporal_transfromer_param"],
            temporal_class_dim=self.opt.sc_params["temporal_class_dim"],
            spatial_conv_levels=self.opt.od_params["spatial_conv_levels"],
            spatial_conv_maps=self.opt.od_params["spatial_conv_maps"],
            spatial_3dconv_layers=self.opt.od_params["spatial_3dconv_layers"],
            spatial_2dconv_layers=self.opt.od_params["spatial_2dconv_layers"],
            spatial_2d_weight=self.opt.od_params["spatial_2d_weight"],
            spatial_3d_weight=self.opt.od_params["spatial_3d_weight"],
            spatial_proj_channels=self.opt.od_params["spatial_proj_channels"],
            spatial_embedding_shape=self.opt.od_params["spatial_embedding_shape"],
            spatial_transfromer_param=self.opt.od_params["spatial_transfromer_param"],
            spatial_num_query=self.opt.od_params["spatial_num_query"],
            spatial_od_dim_list=self.opt.od_params["spatial_od_dim_list"]
        )

    def get_loss_fn(self):
        return opt_fn.spatio_temporal_contrast_loss(
            num_classes=self.model_num_classes,
            seq_length=self.opt.net_params["cubeseq_length"],
            eos_coef=self.opt.data_params["eos_coef"]
        )

    def get_dataloader(self):
        dataset_training = aug.cubic_sequence_data(
            dataset_root=None,
            train_root=['/mnt/nas4/diskm/wangxh/ctca_no/dataset/train_geo_02mm_clean/volumes', '/home/joshua/CAD_diagnosis-master/data/train/labels'],
            pattern='training',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw)
        dataset_testing = aug.cubic_sequence_data(
            dataset_root=None,
            test_root=['/mnt/nas4/diskm/wangxh/ctca_no/dataset/test_geo_02mm_clean/volumes', '/home/joshua/CAD_diagnosis-master/data/test/labels'],
            pattern='testing',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw)
        return DataLoader(dataset_training, batch_size=self.batch_size, shuffle=True, collate_fn=aug.collate_fn),\
               DataLoader(dataset_testing, batch_size=self.batch_size, shuffle=False, collate_fn=aug.collate_fn)


    def pre_training_load(self, ):

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.state_dict_root)

        pretrained_dict_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict_filtered[k] = v

        model_dict.update(pretrained_dict_filtered)
        self.model.load_state_dict(model_dict)