import torch

class LossComputer:
    def __init__(self, criterion, is_robust, dataset, alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset.sens_classes
        _, self.group_counts = dataset.group_counts()
        self.group_counts = self.group_counts.cuda()
        self.group_frac = self.group_counts/self.group_counts.sum()
        #self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        #group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        group_acc, group_count = self.compute_group_avg((yhat > 0.5).float(), group_idx)

        # update historical losses - saves loss for each group - would be interesting to log this
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return group_loss,per_sample_losses.mean(),actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        # add adjustment term (proportional to group size so that small groups get higher loss)
        # here no adjustment is done as adj = 0
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts) # min would be approx adj/3 but on avreage probably around adj/10
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
         # adv probs initiliased as 1/(n_groups)
        # default step size is 0.01. torch.exp(...) = 1 if step_size*adjusted_loss close to 0
        # <  if step_size*adjusted_loss < 0, which is never going to be the case. so i guess it just depends on which ones increase more before they're renormalised
        #  adj loss is probably on average between 0.4 and 1.5
        #so adv probs increase if loss for a certain group is higher
        # step size controls how much to increase adv probs by 
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        # re-normalise adv probs so it sums to 1
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        # multiply each group loss by its adv_probs and sum
        # groups with higher losses will have bigger weighting
        # weighting is also carried over from previous training history
        # so you would only be considering max group loss as total loss if adv probs was 0 for each group except worst performing group
        # eg if adv probs for 1 group was * 5, ie step_size*adjusted_loss=1.6, ie adjust_loss = 160
        # although loss is unbounded, doesn't seem like per sample loss is ever far over 1 (even less for group loss)
        # eg if loss is 1, adv probs is *= 1.01
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss): # not originally in MEDFAIR code
    # ref loss is loss adjusted for sample size for each group
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        # group frac is proportion of people in each group
        # sort group proportions in order of descending loss
        sorted_frac = self.group_frac[sorted_idx]
	
		# think default alpha is 0.2 (TBC)
		# so values in mask would. be true up until cum sum of group proportion exceeds 0.2
        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        # so if sorted_frac = [0.1,0.1,0.5,0.3], weights = [0.5,0.5,0,0,0]
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum() # counts number of true values
        weights[last_idx] = 1 - weights.sum() 
        # maybe this is just to check weights sum to 1, if not gets next weight (eg at index position 2 in my example)
    
        # min var weight is 0 so weights should'nt change
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)
				
        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]

        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float() #size: 2 x batch_size
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        #import pdb; pdb.set_trace()
        
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        # unclear what the point of this is
        # > 0 checks you are only doing it for groups present in the batch
        # if all groups are present prev_weights = array of 0.9 and current weights is array of 0.1
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        # prev loss * 0.9 + current loss * 0.1
        # what is the point of gamma? just weights how much to include current loss in exp avg loss
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        # total samples so far
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float() # prevents 0 division errors
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        # guess this is similar to exp_avg_loss just calculated slightly differently, just based on size of groups (doesn't decay over time i guess)
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()