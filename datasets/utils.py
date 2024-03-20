import numpy as np
import torch
import torchvision.transforms as transforms
import datasets
import pandas as pd
import random
import torchio as tio
from utils.spatial_transforms import ToTensor

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

from torch.utils.data import WeightedRandomSampler


def get_dataset(opt):
    data_setting = opt['data_setting']
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if opt['is_3d']:
        mean_3d = [0.45, 0.45, 0.45]
        std_3d = [0.225, 0.225, 0.225]
        sizes = {'ADNI': (192, 192, 128), 'ADNI3T': (192, 192, 128), 'OCT': (192, 192, 96), 'COVID_CT_MD': (224, 224, 80)}
        if data_setting['augment']:
            transform_train = transforms.Compose([
                tio.transforms.RandomFlip(),
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=15,),
                tio.transforms.CropOrPad(sizes[opt['dataset_name']]),
                
                ToTensor(),
                NormalizeVideo(mean_3d, std_3d),
            ])
        else:
            transform_train = transforms.Compose([
                tio.transforms.CropOrPad(sizes[opt['dataset_name']]),
                ToTensor(),
                NormalizeVideo(mean_3d, std_3d),
            ])
    
        transform_test = transforms.Compose([
            tio.transforms.CropOrPad(sizes[opt['dataset_name']]),
            ToTensor(),
            NormalizeVideo(mean_3d, std_3d),
        ])
    elif opt['is_tabular']:
        pass
    else:
        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(), #anissa
				transforms.RandomRotation((-15, 15)),
                transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(0.3,0.3,0.3), #anissa: brightness saturation, contrast
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)), #anissa, maybe this is too much
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    g = torch.Generator()
    g.manual_seed(opt['random_seed'])
    def seed_worker(worker_id):
        np.random.seed(opt['random_seed'] )
        random.seed(opt['random_seed'])
        
    image_path = data_setting['image_feature_path']

    if opt['dataset_name'] == 'UKBB_RET': # account for having differnet csvs (different binary labels) depending on what prediction task you are doing
        class_name = opt['class_name']
        train_meta = pd.read_csv(data_setting['train_' + class_name+ '_meta_path']) 
        val_meta = pd.read_csv(data_setting['val_' + class_name+ '_meta_path'])
        test_meta = pd.read_csv(data_setting['test_' + class_name+ '_meta_path'])
        print('loaded dataset ', opt['dataset_name'], ' with class name ', class_name)
        if data_setting['adjust_size'] == True:
            train_name = data_setting['train_' + class_name+ '_meta_path']
            val_name = data_setting['val_' + class_name+ '_meta_path']
            test_name = data_setting['test_' + class_name+ '_meta_path']
            
            size = data_setting['dataset_size']
            train_meta = pd.read_csv(train_name.split(".")[0] + str(size) + '.csv')
            val_meta = pd.read_csv(val_name.split(".")[0] + str(size) + '.csv')
            test_meta = pd.read_csv(test_name.split(".")[0] + str(size) + '.csv')
        
            print('got specific csvs of smaller size')
        
        elif data_setting['adjust_centre'] == True:
            # very poor coding from me
            i = data_setting['centre']
            train_meta = pd.read_csv(f'/gpfs3/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR/data/ukbb-ret/splits/train-bp-all-filt2-no-centre{i}.csv')
            val_meta = pd.read_csv(f'/gpfs3/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR/data/ukbb-ret/splits/val-bp-all-filt2-no-centre{i}.csv')
            test_meta = pd.read_csv(f'/gpfs3/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR/data/ukbb-ret/splits/test-bp-all-filt2-no-centre{i}.csv')
        
            print(f'got specific csvs excluding centre {i}')
    else:
        train_meta = pd.read_csv(data_setting['train_meta_path']) 
        val_meta = pd.read_csv(data_setting['val_meta_path'])
        test_meta = pd.read_csv(data_setting['test_meta_path'])   
    
    if opt['is_3d']:
        dataset_name = getattr(datasets, opt['dataset_name'])
        train_data = dataset_name(train_meta, image_path, opt['sensitive_name'], opt['train_sens_classes'], transform_train)
        val_data = dataset_name(val_meta, image_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
        test_data = dataset_name(test_meta, image_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
    elif opt['is_tabular']:
        # different format
        dataset_name = getattr(datasets, opt['dataset_name'])
        data_train_path = data_setting['data_train_path']
        data_val_path = data_setting['data_val_path']
        data_test_path = data_setting['data_test_path']
        
        data_train_df = pd.read_csv(data_train_path)
        data_val_df = pd.read_csv(data_val_path)
        data_test_df = pd.read_csv(data_test_path)
        
        train_data = dataset_name(train_meta, data_train_df, opt['sensitive_name'], opt['train_sens_classes'], None)
        val_data = dataset_name(val_meta, data_val_df, opt['sensitive_name'], opt['sens_classes'], None)
        test_data = dataset_name(test_meta, data_test_df, opt['sensitive_name'], opt['sens_classes'], None)
    
    else:
        dataset_name = getattr(datasets, opt['dataset_name'])
        if  opt['dataset_name'] == 'UKBB_RET' and opt['class_name'] == 'ckd': # different images are saved for this b/c not all images have ckd label
                pickle_train_path = data_setting['pickle_train_ckd_path']
                pickle_val_path = data_setting['pickle_val_ckd_path']
                pickle_test_path = data_setting['pickle_test_ckd_path']
        else:
            pickle_train_path = data_setting['pickle_train_path']
            print('pickle_train_path', pickle_train_path)
            pickle_val_path = data_setting['pickle_val_path']
            pickle_test_path = data_setting['pickle_test_path']
        train_data = dataset_name(train_meta, pickle_train_path, opt['sensitive_name'], opt['train_sens_classes'], transform_train)
        val_data = dataset_name(val_meta, pickle_val_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
        test_data = dataset_name(test_meta, pickle_test_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
    
    print('loaded dataset ', opt['dataset_name'])
        
    if opt['experiment']=='resampling' or opt['experiment']=='GroupDRO' or opt['experiment']=='resamplingSWAD':
        weights = train_data.get_weights(resample_which = opt['resample_which'])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator = g)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=opt['batch_size'], 
                            sampler=sampler,
                            shuffle=(opt['experiment']!='resampling' and opt['experiment']!='GroupDRO' and opt['experiment']!='resamplingSWAD'), num_workers=8, 
                            worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
                          val_data, batch_size=opt['batch_size'],
                          shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
                           test_data, batch_size=opt['batch_size'],
                           shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)

    return train_data, val_data, test_data, train_loader, val_loader, test_loader, val_meta, test_meta
