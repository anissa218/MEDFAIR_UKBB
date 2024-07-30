import parse_args
import json
import numpy as np
import pandas as pd
from utils import basics
import glob
import torch


def train(model, opt):
    for epoch in range(opt['total_epochs']):
        ifbreak = model.train(epoch)
        if ifbreak:
            break
     
    # record val metrics for hyperparameter selection
    pred_df = model.record_val()
    return pred_df
    

if __name__ == '__main__':
    
    opt, wandb = parse_args.collect_args()
    print('pretrained after collect args: ')
    print(opt['pretrained'])
    if not opt['test_mode']:
        
        #random_seeds = np.random.choice(range(100), size = 1, replace=False).tolist() # changed this so so that each model is only trained once!!
        random_seeds = [opt['random_seed']]
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        print('Random seed: ', random_seeds)
        for random_seed in random_seeds:
            print(opt['experiment'])

            # temporary code
            # with open('opt.json', 'w') as f:
            #     json.dump(opt, f, cls=MyEncoder)
            # print('dict saved')

            model = basics.get_model(opt, wandb)

            pred_df = train(model, opt)

            val_df = pd.concat([val_df, pred_df])
            
            pred_df = model.test()
            test_df = pd.concat([test_df, pred_df])
            
        stat_val = basics.avg_eval(val_df, opt, 'val')
        stat_test = basics.avg_eval(test_df, opt, 'test')
        if wandb != None:
            model.log_wandb(stat_val.to_dict())
            model.log_wandb(stat_test.to_dict())        
    else:
        
        if opt['cross_testing']:
            
            test_df = pd.DataFrame()
            if opt['cross_testing_model_path_single'] != '':
                model = basics.get_model(opt, wandb)
                pred_df = model.test()
                test_df = pd.concat([test_df, pred_df])
                print('loaded model from: ', opt['cross_testing_model_path_single'])
                
            else:
                method_model_path = opt['cross_testing_model_path']
                model_paths = glob.glob(method_model_path + '/cross_domain_*.pth')

                for model_path in model_paths:
                    opt['cross_testing_model_path_single'] = model_path
                    model = basics.get_model(opt, wandb)
                    pred_df = model.test()
                    
                    test_df = pd.concat([test_df, pred_df])
            stat_test = basics.avg_eval(test_df, opt, 'cross_testing')
            
            model.log_wandb(stat_test.to_dict())