import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import basemodels
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
from importlib import import_module
from models.GroupDRO.utils import LossComputer
    
    
class GroupDRO(BaseNet):
    def __init__(self, opt, wandb):
        super(GroupDRO, self).__init__(opt, wandb)
        
        self.set_network(opt)
        self.set_optimizer(opt)
        
        self.groupdro_alpha = opt['groupdro_alpha']
        self.groupdro_gamma = opt['groupdro_gamma']
        self.groupdro_step = opt['groupdro_step']
        self.adj = opt['groupdro_adj']
        self.groupdro_step = opt['groupdro_step']

        self.register_buffer("q", torch.ones(self.sens_classes))
        
        self.criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
        #generalization_adjustment = "0"
        #adjustments = [float(c) for c in generalization_adjustment.split(',')]
        adjustments = [self.adj]
        assert len(adjustments) in (1, self.train_data.sens_classes)
        if len(adjustments)==1:
            adjustments = np.array(adjustments* self.train_data.sens_classes)
        else:
            adjustments = np.array(adjustments)
        if self.groupdro_alpha != 1:
            btl=True
        else:
            btl=False
        self.train_loss_computer = LossComputer(
            criterion = self._criterion,
            is_robust=True,
            dataset=self.train_data,
            alpha=self.groupdro_alpha,
            gamma=self.groupdro_gamma,
            adj=adjustments,
            step_size=self.groupdro_step,
            normalize_loss=False,
            btl=btl,
            min_var_weight=0)
    
    def set_network(self, opt):
        """Define the network"""
        
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, in_features= self.in_features, hidden_features = 1024).to(self.device)
        else:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)

    def _train(self, loader):
        """Train the model for one epoch"""
        self.network.train()
        
        running_loss, auc = 0., 0.
        no_iter = 0
        dro_results = {}
        group_losses, mean_losses, losses = [], [], []
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            self.optimizer.zero_grad()
            outputs, features = self.network.forward(images)
            
            group_loss,mean_loss,loss = self.train_loss_computer.loss(outputs, targets, sensitive_attr, is_training = True)
            # group loss is list of losses for each group, mean loss is avg for each sample, loss is robust DRO loss
            group_losses.append(group_loss)
            mean_losses.append(mean_loss)
            losses.append(loss)
            running_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                try:
                    self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})
                except:
                    pass
        dro_results['group_losses'] = group_losses
        dro_results['mean_losses'] = mean_losses
        dro_results['losses'] = losses
        # save dict
        torch.save(dro_results, os.path.join(self.save_path, 'dro_loss_epoch_' + str(self.epoch) + '.pth'))
        
        print('scheduler step')
        self.scheduler.step()
        
        running_loss /= no_iter
        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, running_loss))
        self.epoch += 1
        
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        val_loss, auc = 0., 0.
        no_iter = 0
        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                group_loss,mean_loss,loss = self.train_loss_computer.loss(outputs, targets, sensitive_attr, is_training = False)
                val_loss += loss.item()
                
                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                    
                auc += calculate_auc(outputs.cpu().data.numpy(),
                                               targets.cpu().data.numpy())
                no_iter += 1
                if self.log_freq and (i % self.log_freq == 0):
                    try:
                        self.wandb.log({'Validation loss': val_loss / (i+1), 'Validation AUC': auc / (i+1)})
                    except:
                        pass
    
        auc = 100 * auc / no_iter
        val_loss /= no_iter
        
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        
        return val_loss, auc, log_dict, pred_df  
    
    def _test(self, loader):
        """Compute model output on testing set"""

        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            feature_vectors = []

            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                feature_vectors.append(features.to('cpu'))

                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
        
        # save features from test inference
        feature_tensor = torch.cat(feature_vectors)
        torch.save(feature_tensor, os.path.join(self.save_path, 'features.pt'))
        index_tensor = torch.tensor(tol_index)
        torch.save(index_tensor, os.path.join(self.save_path, 'index.pt'))
        print('saved features')

        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        pred_df.to_csv(os.path.join(self.save_path, self.experiment + 'pred.csv'), index = False)
        #basics.save_results(t_predictions, tol_target, s_prediction, tol_sensitive, self.save_path)
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
            
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')

        return log_dict