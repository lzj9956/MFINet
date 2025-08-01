def get_config():
    class General:
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_size_per_gpu = 6
        fp16 = True
        SeqDir = '/home/chx/Work/semantic-kitti/sequences'
        category_list = ["car", "bicycle", "motorcycle", "truck",
                      "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
                      "parking", "sidewalk", "other-ground", "building", "fence",
                      "vegetation", "trunk", "terrain", "pole", "traffic-sign"]
        loss_mode = 'wce'
        class Voxel:
            RV_theta = (-25.0, 3.0)
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-4.0, 2.0)

            bev_shape = (600, 600, 30)
            rv_shape = (64, 2048)

    class DatasetParam:
        class Train:
            data_src = 'data'
            num_workers = 6
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            class CopyPasteAug:
                is_use = False
                ObjBackDir = 'object_bank_semkitti'
                paste_max_obj_num = 20
            class AugParam:
                noise_mean = 0
                noise_std = 0.0001
                theta_range = (-180.0, 180.0)
                shift_range = ((-3, 3), (-3, 3), (-0.4, 0.4))
                size_range = (0.95, 1.05)

        class Val:
            data_src = 'data'
            num_workers = 8
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel

    class ModelParam:
        prefix = "bev_only.AttNet"
        Voxel = General.Voxel
        category_list = General.category_list
        class_num = len(category_list) + 1
        loss_mode = General.loss_mode

        point_feat_out_channels = 64
        fusion_mode = 'MLPFusion'
        fusion_way = 'Cat'

        class BEVParam:
            base_block = 'BasicBlock'
            context_layers = [64, 32, 64, 128]
            layers = [2, 3, 4]
            bev_grid2point = dict(type='BilinearSample', scale_rate=(0.5, 0.5))

        class pretrain:
            pretrain_epoch = 52


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            base_lr = 0.00002
            momentum = 0.9
            nesterov = True
            wd = 1e-3

        class schedule:
            type = "cosine"
            begin_epoch = 0
            end_epoch = 48
            max_lr = 0.02

    return General, DatasetParam, ModelParam, OptimizeParam