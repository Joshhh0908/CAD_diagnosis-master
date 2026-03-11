class DefaultConfig(object):

    data_params = {
        "dataset_root": '/mnt/nas4/diskm/wangxh/CTCA_handover_26_allbranch_02to04mm_revise4_disk4/dataset/train_geo_02mm_clean/',
        "eos_coef": 0.2,
        "train_ratio": 0.8,
        "window_lw": [300, 900],
        "batch_size": 2
    }
    net_params = {
        "input_shape": [256, 64, 64],
        "cubeseq_length": 32,
        "num_classes": [3, 6],
        "ret_map": True,
        "in_channels": 1
    }
    sc_params = {
        "_3d_cube_selection": [32, 25, 8],
        "temporal_conv_levels": 4,
        "temporal_conv_maps": [16, 32, 64, 128],
        "temporal_feature_channels": [128, 32],
        "temporal_embedding_dim": [864, 512],
        "temporal_transfromer_param": [8, 4],
        "temporal_class_dim": [512, 128]
    }
    od_params = {
        "spatial_conv_levels" : 4,
        "spatial_conv_maps" : [16, 32, 64, 128],
        "spatial_3dconv_layers" : [2, 2, 3, 3],
        "spatial_2dconv_layers" : [2, 2, 2, 2],
        "spatial_2d_weight" : [0.25, 0.25, 0.25, 0.25],
        "spatial_3d_weight" : 0.75,
        "spatial_proj_channels" : [128, 1024, 128, 512],
        "spatial_embedding_shape" : [16, 512],
        "spatial_transfromer_param" : [4, 4],
        "spatial_num_query" : 16,
        "spatial_od_dim_list" : [512, 256]
    }


opt = DefaultConfig()

class placeholder_cfg:
    def __init__(self, H, W, Z):
        self.H = H
        self.W = W
        self.Z = Z

cfg = placeholder_cfg(40,40,480)