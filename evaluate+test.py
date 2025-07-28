import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True


def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]

def val_fp16(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    print('FP16 inference mode!')
    model.eval()
    f = open(os.path.join(save_path, 'record_fp16_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, fname_pcd, path_seq, path_name, learning_map) in tqdm.tqdm(enumerate(val_loader)):
            with torch.cuda.amp.autocast():
                pred_cls = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), pcds_sphere_coord.squeeze(0).cuda())
                # pred_cls = model.infer(pcds_xyzi.cuda(), pcds_coord.cuda(), pcds_sphere_coord.cuda())
            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pred_cls_argmax = pred_cls.argmax(dim=1)
            # pcds_target = pcds_target[0, :, 0].contiguous()

            path_seq = path_seq[0]
            path_name = path_name[0]

            # 保存预测结果
            pred_np = pred_cls_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            pred_np = map(pred_np, learning_map)

            # save scan
            path = os.path.join("sequences", path_seq, "predictions", path_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pred_np.tofile(path)



def val(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    
    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, fname_pcd) in tqdm.tqdm(enumerate(val_loader)):
            # pred_cls = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), pcds_sphere_coord.squeeze(0).cuda())
            pred_cls = model.infer(pcds_xyzi.cuda(), pcds_coord.cuda(),pcds_sphere_coord.cuda())
            
            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = pcds_target[0, :, 0].contiguous()
            
            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])
        
        #record segmentation metric
        metric_cate = criterion_cate.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
        
        f.write(string + '\n')
        f.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")


    # define dataloader
    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)
    
    # define model
    model = eval(pModel.prefix)(pModel)
    model.cuda()
    model.eval()


    for epoch in range(args.start_epoch, args.end_epoch + 1):
        pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch))
        print(pretrain_model)
        checkpoint = torch.load(pretrain_model, map_location='cuda:0')  # 加载模型参数到单个GPU
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if pGen.fp16:
            val_fp16(epoch, model, val_loader, pGen.category_list, save_path, 0)  # 在单GPU上进行验证
        else:
            val(epoch, model, val_loader, pGen.category_list, save_path, 0)  # 在单GPU上进行验证

def seed_torch(seed=1024):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = False
    print("We use the seed: {}".format(seed))

if __name__ == '__main__':
    seed_torch(seed=520)
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', default='config/wce.py', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--start_epoch', type=int, default=46)
    parser.add_argument('--end_epoch', type=int, default=46)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)