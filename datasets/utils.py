import numpy as np
import torch
import torchvision.transforms as transforms
import datasets
import pandas as pd
import random
import torchio as tio
from utils.spatial_transforms import ToTensor
import os

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
    elif opt['is_small']:
        print('is small')
        if data_setting['augment']:
            transform_train = transforms.Compose([
                #transforms.Resize(28),
                transforms.ToTensor(), # tried putting ToTEnsor first
                transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(), #anissa
				transforms.RandomRotation((-15, 15)),
                #transforms.RandomCrop((28, 28)), # effectively no random cropping
                #transforms.ColorJitter(0.3,0.3,0.3), #anissa: brightness saturation, contrast
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)), #anissa, maybe this is too much
                transforms.Normalize(mean=[0.45],std=[0.225]),
            ])
        else:
            transform_train = transforms.Compose([
                #transforms.Resize(28),
                #transforms.CenterCrop(28),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.45],std=[0.225]),
            ])
    
        transform_test = transforms.Compose([
            #transforms.Resize(28),
            #transforms.CenterCrop(28),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.45],std=[0.225]),
        ])
    else:
        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.Resize(299),
                transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(), #anissa
				transforms.RandomRotation((-15, 15)),
                #transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(0.3,0.3,0.3), #anissa: brightness saturation, contrast
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)), #anissa, maybe this is too much
                transforms.ToTensor(),
                normalize,
            ])
        else:
            print('no augmentation')
            transform_train = transforms.Compose([
                transforms.Resize(299),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
        transform_test = transforms.Compose([
            transforms.Resize(299),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    g = torch.Generator()
    g.manual_seed(opt['random_seed'])
    def seed_worker(worker_id):
        np.random.seed(opt['random_seed'] )
        random.seed(opt['random_seed'])
        
    image_path = data_setting['image_feature_path']

    if opt['dataset_name'] == 'UKBB_RET' or opt['dataset_name'] == 'MNIST': # account for having differnet csvs (different binary labels) depending on what prediction task you are doing
        
        class_name = opt['class_name']
        if class_name: # empty strings are evaluated to falsy
            class_name = '_' + class_name # this should mean that for MNIST dataset it still works if class_name is empty
        root_path = data_setting['root_path']
        data_folder = opt['data_folder'] # to implement!

        train_meta = pd.read_csv(os.path.join(root_path,data_folder,data_setting['train' + class_name+ '_meta_path'])) 
        val_meta = pd.read_csv(os.path.join(root_path,data_folder,data_setting['val' + class_name+ '_meta_path'])) 
        test_meta = pd.read_csv(os.path.join(root_path,data_folder,data_setting['test' + class_name+ '_meta_path'])) 

        print('loaded dataset ', opt['dataset_name'], 'with folder name ', data_folder, ' and with class name ', class_name)
        

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
        if  opt['dataset_name'] == 'UKBB_RET' or opt['dataset_name'] == 'MNIST': # different images are saved for this b/c not all images have ckd label
            pickle_train_path = os.path.join(root_path,data_folder,data_setting['pickle_train_path'])
            pickle_val_path = os.path.join(root_path,data_folder,data_setting['pickle_val_path'])
            pickle_test_path = os.path.join(root_path,data_folder,data_setting['pickle_test_path'])
            
            print('pickle_train_path', pickle_train_path)

        else:
            pickle_train_path = data_setting['pickle_train_path']
            print('pickle_train_path', pickle_train_path)
            pickle_val_path = data_setting['pickle_val_path']
            pickle_test_path = data_setting['pickle_test_path']
        
        # so that pickle files are used in default case unless training splits are different eg with adjust centre 
        if data_setting['adjust_centre'] == True or data_setting['adjust_size'] == True:
            train_data = dataset_name(train_meta, pickle_train_path, opt['sensitive_name'], opt['train_sens_classes'], transform_train,use_pkl=False)
            val_data = dataset_name(val_meta, pickle_val_path, opt['sensitive_name'], opt['sens_classes'], transform_test,use_pkl=False)
            test_data = dataset_name(test_meta, pickle_test_path, opt['sensitive_name'], opt['sens_classes'], transform_test,use_pkl=False)
        else:
            train_data = dataset_name(train_meta, pickle_train_path, opt['sensitive_name'], opt['train_sens_classes'], transform_train)
            val_data = dataset_name(val_meta, pickle_val_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
            test_data = dataset_name(test_meta, pickle_test_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
    
    print('loaded dataset ', opt['dataset_name'])
        
    if opt['experiment']=='resampling' or opt['experiment']=='resamplingSWAD': # or opt['experiment']=='GroupDRO' 
        print('resampling')
        weights = train_data.get_weights(resample_which = opt['resample_which'])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator = g)
        print('weights', list(set(weights)))
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=opt['batch_size'], 
                            sampler=sampler,
                            shuffle=(opt['experiment']!='resampling' and opt['experiment']!='resamplingSWAD'), num_workers=8, 
                            worker_init_fn=seed_worker, generator=g, pin_memory=True) # for shuffle - and opt['experiment']!='GroupDRO' 
    val_loader = torch.utils.data.DataLoader(
                          val_data, batch_size=opt['batch_size'],
                          shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
                           test_data, batch_size=opt['batch_size'],
                           shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)

    return train_data, val_data, test_data, train_loader, val_loader, test_loader, val_meta, test_meta
