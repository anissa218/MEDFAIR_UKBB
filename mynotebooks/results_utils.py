import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.manifold import TSNE


### PRE PROCESSING FUNCTIONS ###
def find_threshold_anissa(tol_output, tol_target):

    PRED_LABEL = ['disease']

    # create empty dfs
    thrs = []            
        
    for j in range(0, len(tol_output)):
        thisrow = {}
        truerow = {}

        # iterate over each entry in prediction vector; each corresponds to
        # individual label
        for k in range(len(PRED_LABEL)):
            thisrow["prob_" + PRED_LABEL[k]] = tol_output[j]
            truerow[PRED_LABEL[k]] = tol_target[j]
           
    for column in PRED_LABEL:
        
        thisrow = {}
        thisrow['label'] = column
        thisrow['bestthr'] = np.nan

        p, r, t = sklm.precision_recall_curve(tol_target, tol_output)
        p =p[:-1] # remove last precision and recall value bc sklearn just adds them for y axis alignment of graph (doesn't correspond to a threshold)
        r = r[:-1]

        tnrs=[]

        for threshold in t:
            tol_output_binary = np.where(tol_output > threshold, 1, 0)
            tnr = (((tol_output_binary == 0) & (tol_target == 0)).sum())/((tol_target == 0).sum())
            tnrs.append(tnr)

        # Choose the best threshold based on the highest F1 measure
        f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
        score = f1+np.multiply(0.25,tnrs)

        bestthr = t[np.where(score == max(score))]
        thrs.append(bestthr)
        
        thisrow['bestthr'] = bestthr[0]

    return bestthr[0]

def preprocess_mimic_data(path_to_preds,path_to_splits):
    '''
    Function to merge the predictions dataframe with the metadata (specific to mimic).
    Also add extra columns for FPs and FNs.
    
    Args:
    path_to_preds: path to the predictions file
    path_to_splits: path to the csv file with all the metadata for each split

    Returns:
    test_df: dataframe with the predictions and metadata and some stats
    '''
    pred_df = pd.read_csv(path_to_preds)
    pred_df = pred_df.set_index('index')
    test_df = pd.read_csv(path_to_splits)
    test_df['binary_label'] = np.float64(test_df['No Finding'].astype(float) > 0)

    # add metadata
    metadata_path = '/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR/data/other_data/mimic-cxr/physionet.org/files/mimiciv/2.2/hosp/'
    admissions_file = 'admissions.csv.gz'

    admissions_df = pd.read_csv(os.path.join(metadata_path,admissions_file))
    admissions_df_no_duplicates = admissions_df.drop_duplicates(subset='subject_id')
    admissions_df_no_duplicates = admissions_df_no_duplicates[['subject_id','hadm_id','insurance','language','marital_status','race']] # keep only relevant columns
    test_df= pd.merge(test_df,admissions_df_no_duplicates[['subject_id','insurance', 'marital_status']], on='subject_id',how='left')

    # add extra stats on FPs and FNs
    test_df['pred'] = pred_df['pred']

    if 'raw_pred' in pred_df.columns: # for models where i also saved raw preds
        test_df['raw_pred'] = pred_df['raw_pred']

    # inverted these because 0 is disease and 1 is no disease (negatiev)
    test_df['FN'] = np.where((test_df['binary_label']==0) & (test_df['pred']==1),1,0)
    test_df['FP'] = np.where((test_df['binary_label']==1) & (test_df['pred']==0),1,0)
    test_df['TN'] = np.where((test_df['binary_label']==1) & (test_df['pred']==1),1,0)
    test_df['TP'] = np.where((test_df['binary_label']==0) & (test_df['pred']==0),1,0)

    # add procedure information
    ROOT_FOLDER = '/gpfs3/well/papiez/shared/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0'
    metadata_path = os.path.join(ROOT_FOLDER, 'mimic-cxr-2.0.0-metadata.csv.gz')
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['image'] = metadata_df['dicom_id']+'.jpg'
    test_df= pd.merge(test_df,metadata_df[['subject_id','study_id','image','PerformedProcedureStepDescription']], on=['subject_id','study_id','image'],how='left')

    # add an extra category for ethnicity
    test_df['Race_cat'] = test_df['Race_multi'].apply(lambda x: 'White' if x=='WHITE' else 
                                       'White-other' if 'WHITE -' in x else 
                                       'Hispanic' if 'HISPANIC' in x or 'LATINO' in x or 'MEXICAN' in x or 'CENTRAL AMERICAN' in x or 'CUBAN' in x or 'SALVADORAN' in x or 'GUATEMALAN' in x or 'DOMINICAN' in x or 'COLOMBIAN' in x or 'HONDURAN' in x else 
                                       'Asian' if 'ASIAN' in x or 'CHINESE' in x or 'KOREAN' in x or 'SOUTH EAST ASIAN' in x or 'ASIAN INDIAN' in x else 
                                       'Black' if 'BLACK' in x or 'AFRICAN AMERICAN' in x or 'CARIBBEAN ISLAND' in x or 'CAPE VERDEAN' in x else 
                                       'Other')

    return test_df

def preprocess_ukbb_data(path_to_preds,path_to_splits):
    '''
    Function to merge the predictions dataframe with the metadata (specific to ukbb).
    Also add extra columns for FPs and FNs.
    
    Args:
    path_to_preds: path to the predictions file
    path_to_splits: path to the csv file with all the metadata for each split

    Returns:
    metadata_df: dataframe with the predictions and metadata and some stats
    '''
    pred_df = pd.read_csv(path_to_preds)
    pred_df = pred_df.set_index('index')
    metadata_df = pd.read_csv(path_to_splits)
    metadata_df['binary_label'] = metadata_df['binaryLabel'].astype(float)

    # add predictions from pred file
    metadata_df['pred'] = pred_df['pred']
    metadata_df['test_label'] = pred_df['label'].astype(float)
    metadata_df['raw_pred'] = pred_df['raw_pred']

    if (metadata_df['binary_label'] == metadata_df['test_label']).all() == False:
        print('prediction label and CSV label do not match')
        return

    else:
        metadata_df.drop('test_label',axis=1,inplace=True)

    # modify pred with new threshold
    metadata_df['old_pred'] = metadata_df['pred']
    threshold = find_threshold_anissa(metadata_df['raw_pred'],metadata_df['binary_label'])
    metadata_df['pred'] = (metadata_df['raw_pred'] > threshold).astype(int)

    # add extra stats on FPs and FNs
    metadata_df['FP'] = np.where((metadata_df['binary_label']==0) & (metadata_df['pred']==1),1,0)
    metadata_df['FN'] = np.where((metadata_df['binary_label']==1) & (metadata_df['pred']==0),1,0)
    metadata_df['TP'] = np.where((metadata_df['binary_label']==1) & (metadata_df['pred']==1),1,0)
    metadata_df['TN'] = np.where((metadata_df['binary_label']==0) & (metadata_df['pred']==0),1,0)

    # add bmi_cat column
    metadata_df['bmi_cat']=pd.cut(metadata_df['bmi_at_imaging'], bins=[0,24,26.5,29.5,100], labels=[0,1,2,3])

    # could discretise this more (not just binary)
    metadata_df['high_bp'] = metadata_df.apply(lambda x: 1 if x.loc['4079'] >= 90 and x.loc['4080'] >= 140 else 0, axis=1)

    metadata_df['deprivation_index']=pd.cut(metadata_df['26410'], bins=[0,8,13,23,100], labels=[0,1,2,3])

    metadata_df['ethnicity'] = metadata_df['21000'].astype(str).str[0]
    #metadata_df['ethnicity']=metadata_df['ethnicity'].astype(int)
    metadata_df['gen_ethnicity'] = metadata_df['22006'].apply(lambda x: 1 if x==1 else 0)
    metadata_df.rename(columns={'54':'assessment_centre'}, inplace=True)
    metadata_df.rename(columns={'1558':'alcohol'}, inplace=True)
    metadata_df.rename(columns={'22032':'physical_activity'}, inplace=True)

    return metadata_df

def preprocess_mnist_data(path_to_preds,path_to_splits):
     '''
    Function to merge the predictions dataframe with the metadata (specific to MNIST).
    Also add extra columns for FPs and FNs.
    
    Args:
    path_to_preds: path to the predictions file
    path_to_splits: path to the csv file with all the metadata for each split

    Returns:
    metadata_df: dataframe with the predictions and metadata and some stats
    '''
     pred_df = pd.read_csv(path_to_preds)
     pred_df = pred_df.set_index('index')
     metadata_df = pd.read_csv(path_to_splits)
     metadata_df['binary_label'] = metadata_df['binaryLabel'].astype(float)

    # add predictions from pred file
     metadata_df['pred'] = pred_df['pred']
     metadata_df['test_label'] = pred_df['label'].astype(float)
     metadata_df['raw_pred'] = pred_df['raw_pred']
     
     if (metadata_df['binary_label'] == metadata_df['test_label']).all() == False:
        print('prediction label and CSV label do not match')
        return
     else:
        metadata_df.drop('test_label',axis=1,inplace=True)
    # modify pred with new threshold
     metadata_df['old_pred'] = metadata_df['pred']
     threshold = find_threshold_anissa(metadata_df['raw_pred'],metadata_df['binary_label'])
     metadata_df['pred'] = (metadata_df['raw_pred'] > threshold).astype(int)
     
     metadata_df['FP'] = np.where((metadata_df['binary_label']==0) & (metadata_df['pred']==1),1,0)
     metadata_df['FN'] = np.where((metadata_df['binary_label']==1) & (metadata_df['pred']==0),1,0)
     metadata_df['TP'] = np.where((metadata_df['binary_label']==1) & (metadata_df['pred']==1),1,0)
     metadata_df['TN'] = np.where((metadata_df['binary_label']==0) & (metadata_df['pred']==0),1,0)

     return metadata_df

    
def get_val_train_preds(path_to_results_folder,path_to_train_splits,path_to_val_splits,preprocessing_function=preprocess_ukbb_data,pretrained=True):

    '''
    Function to get the predictions for the validation and training sets.
    args:
    path_to_results_folder: path to the folder where the predictions are saved for all epochs
    path_to_train_splits: path to the csv file with all the metadata for all training data
    path_to_val_splits: path to the csv file with all the metadata for all validation data
    preprocessing_function: function to preprocess the data (specific to the dataset)
    Returns:
    train_preds: dictionary with the training predictions
    val_preds: dictionary with the validation predictions
    '''

    root_path = path_to_results_folder
    file_list = os.listdir(root_path)
    val_preds = {}
    train_preds = {}

    if pretrained==True:

        pattern = re.compile(r'pretrained_epoch_(\d+)_([a-z]+)_pred.csv')
    else:
        pattern = re.compile(r'not_pretrained_epoch_(\d+)_([a-z]+)_pred.csv')

    for file_name in file_list:
        match = pattern.match(file_name)
        if match:
            epoch_num = int(match.group(1))
            dataset_type = match.group(2)
                
            if dataset_type == 'val':
                val_preds[epoch_num-1] = file_name # for some reason the val preds are one epoch ahead
            elif dataset_type == 'train':
                train_preds[epoch_num] = file_name

    for epoch_num, file_name in val_preds.items():
        path_to_preds = os.path.join(root_path,file_name)
        path_to_splits = path_to_val_splits
        val_preds[epoch_num]= preprocessing_function(path_to_preds,path_to_splits) # replace the file name with the processed dataframe

    for epoch_num, file_name in train_preds.items():
        path_to_preds = os.path.join(root_path,file_name)
        path_to_splits = path_to_train_splits
        train_preds[epoch_num]= preprocessing_function(path_to_preds,path_to_splits) # replace the file name with the processed dataframe
        
    return train_preds,val_preds

### ANALYSIS FUNCTIONS ###

def get_overall_auc(train_preds,val_preds):
    train_auc, val_auc = [],[]
    for epoch_num in range(len(val_preds)):
        val_df = val_preds[epoch_num]
        train_df = train_preds[epoch_num]
        train_auc.append(roc_auc_score(train_df['binary_label'], train_df['raw_pred']))
        val_auc.append(roc_auc_score(val_df['binary_label'], val_df['raw_pred']))

    return train_auc,val_auc

def get_overall_metrics(train_preds,val_preds):
    val_acc = []
    train_acc = []
    val_precision = []
    train_precision = []
    val_recall = []
    train_recall = []
    train_tnr = []
    val_tnr = []
    for epoch_num in range(len(val_preds)):
        val_df = val_preds[epoch_num]
        train_df = train_preds[epoch_num]
        val_acc.append(len(val_df[val_df['pred']==val_df['binary_label']])/len(val_df))
        train_acc.append(len(train_df[train_df['pred']==train_df['binary_label']])/len(train_df))
        val_precision.append(len(val_df[val_df['TP']==1])/(len(val_df[val_df['TP']==1])+len(val_df[val_df['FP']==1])))
        train_precision.append(len(train_df[train_df['TP']==1])/(len(train_df[train_df['TP']==1])+len(train_df[train_df['FP']==1])))
        val_recall.append(len(val_df[val_df['TP']==1])/(len(val_df[val_df['TP']==1])+len(val_df[val_df['FN']==1])))
        train_recall.append(len(train_df[train_df['TP']==1])/(len(train_df[train_df['TP']==1])+len(train_df[train_df['FN']==1])))
        train_tnr.append(len(train_df[train_df['TN']==1])/(len(train_df[train_df['TN']==1])+len(train_df[train_df['FP']==1])))
        val_tnr.append(len(val_df[val_df['TN']==1])/(len(val_df[val_df['TN']==1])+len(val_df[val_df['FP']==1])))

    return train_acc,val_acc,train_precision,val_precision,train_recall,val_recall, train_tnr, val_tnr

def filter_groups(group,min_size):
    # to make sure subgroups are of a minimum size
    return len(group) > min_size

def get_stats(df,col_name,filter_group_size=True,min_size = 0.01):
    
    grouped_df = df.groupby(col_name)
    threshold = min_size*len(df)

    if filter_group_size:
        mask = grouped_df.apply(filter_groups,min_size=threshold)
    else:
        mask = grouped_df.apply(filter_groups,min_size=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    
        accuracy = grouped_df.apply(lambda group: (group['binary_label'] == group['pred']).mean())[mask]
        precision = grouped_df.apply(lambda group: (group['TP']).sum()/((group['TP']).sum()+(group['FP']).sum()))[mask]
        recall = grouped_df.apply(lambda group: (group['TP']).sum()/((group['FN']).sum()+(group['TP']).sum()))[mask]
        n_groups = grouped_df.apply(lambda group: len(group))[mask]
        pos_labels = grouped_df.apply(lambda group: (1 - group['binary_label']).mean())[mask] # this is when 1=no finding, 0 = finding, so need to swap mean around
        tnr = grouped_df.apply(lambda group: (group['TN']).sum()/((group['TN']).sum()+(group['FP']).sum()))[mask]

    
    return accuracy, precision, recall, n_groups, pos_labels, tnr


def get_subgroup_auc(preds,col_name,filter_group_size=True,min_size = 0.01):
    train_preds = preds
    col_name = col_name
    train_auc_list = []
    for epoch_num in range(len(train_preds)):
        df = train_preds[epoch_num]

        grouped_df = df.groupby(col_name)
        threshold = min_size*len(df)
        filtered_groups = grouped_df.filter(lambda x: len(x) >= threshold) # in AUC calculation need to filter regardless because can't have same label for all members of a group
        filtered_groups = filtered_groups.groupby(col_name) # not sure why but apparently need to do this again

        auc = filtered_groups.apply(lambda group:roc_auc_score(group['binary_label'], group['raw_pred']))

        train_auc_list.append(auc)
        
    return train_auc_list

def get_subgroup_positive_preds(preds,col_name,filter_group_size=True,min_size = 0.01):
    train_preds = preds
    col_name = col_name
    train_pp_list = []
    for epoch_num in range(len(train_preds)):
        df = train_preds[epoch_num]

        grouped_df = df.groupby(col_name)
        threshold = min_size*len(df)
        filtered_groups = grouped_df.filter(lambda x: len(x) >= threshold) # in AUC calculation need to filter regardless because can't have same label for all members of a group
        filtered_groups = filtered_groups.groupby(col_name) # not sure why but apparently need to do this again

        pp = filtered_groups.apply(lambda group:group['pred'].mean())

        train_pp_list.append(pp)
        
    return train_pp_list

def get_subgroup_stats(preds,col_name):
    '''
    Function to analyse same metrics but for each subgroup
    '''
    train_preds = preds
    col_name = col_name
    train_acc_list,train_precision_list, train_recall_list,train_f1_list,train_tnr_list= [],[],[],[],[]

    for epoch_num in range(len(train_preds)): # should probably fine a more elegant way to do this
        # train data:
        accuracy, precision, recall, n_groups, pos_labels,tnr = get_stats(train_preds[epoch_num],col_name)
        f1 = 2*(precision*recall)/(precision+recall)

        train_acc_list.append(accuracy)
        train_precision_list.append(precision)
        train_recall_list.append(recall)
        train_f1_list.append(f1)
        train_tnr_list.append(tnr)
    
    return train_acc_list,train_precision_list, train_recall_list,train_f1_list, train_tnr_list


def make_results_df(train_preds,val_preds,subgroups):
    '''
    Function for quantitative comparison. Still need to add AUC
    Just looks at results at final epoch (5th to last one actually!)
    '''
    model_results_df = pd.DataFrame(columns=['Subgroup','Train Acc', 'Train Acc Gap','Val Acc','Val Acc Gap','Train Precision','Train Precision Gap','Val Precision','Val Precision Gap','Train Recall','Train Recall Gap','Val Recall','Val Recall Gap'])

    for subgroup in subgroups:
        train_acc_list,train_precision_list, train_recall_list,train_f1_list,train_tnr_list = get_subgroup_stats(train_preds,subgroup)
        val_acc_list, val_precision_list, val_recall_list,val_f1_list, val_tnr_list = get_subgroup_stats(val_preds,subgroup)
        # # AUC only works if you have enough examples of each class
        train_auc_list = get_subgroup_auc(train_preds,subgroup)
        val_auc_list = get_subgroup_auc(val_preds,subgroup)

        # changed to -5 because since early stopping is 5 it's technically at this point that model is used
        train_acc,train_precision, train_recall,train_f1,train_tnr = train_acc_list[-5],train_precision_list[-5], train_recall_list[-5],train_f1_list[-5],train_tnr_list[-5]
        val_acc,val_precision, val_recall,val_f1,val_tnr = val_acc_list[-5], val_precision_list[-5], val_recall_list[-5],val_f1_list[-5], val_tnr_list[-5]
        gap_train_acc,gap_train_precision,gap_train_recall,gap_train_tnr = train_acc.max()-train_acc.min(),train_precision.max()-train_precision.min(),train_recall.max()-train_recall.min(), train_tnr.max()-train_tnr.min()
        gap_val_acc,gap_val_precision,gap_val_recall, gap_val_tnr = val_acc.max()-val_acc.min(),val_precision.max()-val_precision.min(),val_recall.max()-val_recall.min(), val_tnr.max()-val_tnr.min()
        min_train_acc,min_train_precision,min_train_recall,min_train_tnr = train_acc.min(),train_precision.min(),train_recall.min(),train_tnr.min()
        min_val_acc,min_val_precision,min_val_recall,min_val_tnr = val_acc.min(),val_precision.min(),val_recall.min(),val_tnr.min()


        train_acc,val_acc,train_precision,val_precision,train_recall,val_recall,train_tnr,val_tnr=get_overall_metrics(train_preds,val_preds) # overall metrics, independent of subgroup
        train_acc,val_acc,train_precision,val_precision,train_recall,val_recall,train_tnr,val_tnr = train_acc[-5],val_acc[-5],train_precision[-5],val_precision[-5],train_recall[-5],val_recall[-5],train_tnr[-5],val_tnr[-5]  # only get last epoch

        train_auc,val_auc = train_auc_list[-5],val_auc_list[-5]
        gap_train_auc,gap_val_auc = train_auc.max()-train_auc.min(),val_auc.max()-val_auc.min()
        min_train_auc,min_val_auc = train_auc.min(),val_auc.min()

        overall_train_auc,overall_val_auc = get_overall_auc(train_preds,val_preds)
        overall_train_auc,overall_val_auc = overall_train_auc[-5],overall_val_auc[-5]

        #model_results_df = model_results_df.append({'Subgroup':subgroup,'Train Acc':train_acc,'Train Acc Gap':gap_train_acc,'Val Acc':val_acc,'Val Acc Gap':gap_val_acc,'Train Precision':train_precision,'Train Precision Gap':gap_train_precision,'Val Precision':val_precision,'Val Precision Gap':gap_val_precision,'Train Recall':train_recall,'Train Recall Gap':gap_train_recall,'Val Recall':val_recall,'Val Recall Gap':gap_val_recall}, ignore_index = True) #'Train AUC':train_auc, 'Train AUC Gap': gap_train_auc, 'Val AUC': val_auc, 'Val AUC Gap': gap_val_auc},ignore_index=True)
    
        new_row = pd.DataFrame({'Subgroup': [subgroup],
                        'Train Acc': [train_acc],
                        'Train Acc Gap': [gap_train_acc],
                        'Train Min Acc': [min_train_acc],
                        'Val Acc': [val_acc],
                        'Val Acc Gap': [gap_val_acc],
                        'Val Min Acc': [min_val_acc],
                        'Train AUC': [overall_train_auc], 
                       'Train AUC Gap': [gap_train_auc], 
                       'Train Min AUC': [min_train_auc],
                       'Val AUC': [overall_val_auc], 
                       'Val AUC Gap': [gap_val_auc],
                       'Val Min AUC': [min_val_auc],
                        'Train Precision': [train_precision],
                        'Train Precision Gap': [gap_train_precision],
                        'Train Min Precision': [min_train_precision],
                        'Val Precision': [val_precision],
                        'Val Precision Gap': [gap_val_precision],
                        'Val Min Precision': [min_train_precision],
                        'Train Recall': [train_recall],
                        'Train Recall Gap': [gap_train_recall],
                        'Train Min Recall': [min_train_recall],
                        'Val Recall': [val_recall],
                        'Val Recall Gap': [gap_val_recall],
                        'Val Min Recall': [min_val_recall],
                        'Train TNR': [train_tnr],
                        'Train TNR Gap': [gap_train_tnr],
                        'Train Min TNR': [min_train_tnr],
                        'Val TNR': [val_tnr],
                        'Val TNR Gap': [gap_val_tnr],
                        'Val Min TNR': [min_val_tnr]}, 
                       index=[0])

        model_results_df = pd.concat([model_results_df, new_row], ignore_index=True)

    return model_results_df

def make_test_results_df(train_preds,subgroups):
    '''
    Function for quantitative comparison.
    Could definitely optimise
    '''
    model_results_df = pd.DataFrame()

    for subgroup in subgroups:
        train_acc_list,train_precision_list, train_recall_list,train_f1_list,train_tnr_list = get_subgroup_stats(train_preds,subgroup)
        # # AUC only works if you have enough examples of each class
        train_auc = get_subgroup_auc(train_preds,subgroup)[-1]
        train_acc,train_precision, train_recall,train_f1, train_tnr = train_acc_list[-1],train_precision_list[-1], train_recall_list[-1],train_f1_list[-1], train_tnr_list[-1]
        gap_train_acc,gap_train_precision,gap_train_recall, gap_train_tnr = train_acc.max()-train_acc.min(),train_precision.max()-train_precision.min(),train_recall.max()-train_recall.min(), train_tnr.max()-train_tnr.min()
        
        gap_train_auc = train_auc.max()-train_auc.min()

        min_acc,min_auc,min_precision,min_recall,min_tnr = train_acc.min(),train_auc.min(),train_precision.min(),train_recall.min(),train_tnr.min()


        train_acc,val_acc,train_precision,val_precision,train_recall,val_recall,train_tnr,val_tnr=get_overall_metrics(train_preds,train_preds) # overall metrics, independent of subgroup
        train_acc,val_acc,train_precision,val_precision,train_recall,val_recall,train_tnr,val_tnr = train_acc[-1],val_acc[-1],train_precision[-1],val_precision[-1],train_recall[-1],val_recall[-1],train_tnr[-1],val_tnr[-1] # only get last epoch
        train_auc,val_auc = get_overall_auc(train_preds,train_preds)
        train_auc = train_auc[-1]
    
        new_row = pd.DataFrame({'Subgroup': [subgroup],
                        'Test Acc': [train_acc],
                        'Test Acc Gap': [gap_train_acc],
                        'Test Min Acc': [min_acc],
                        'Test AUC': [train_auc],
                        'Test AUC Gap': [gap_train_auc],
                        'Test Min AUC': [min_auc],
                        'Test Precision': [train_precision],
                        'Test Precision Gap': [gap_train_precision],
                        'Test Min Precision': [min_precision],
                        'Test Recall': [train_recall],
                        'Test Recall Gap': [gap_train_recall],
                        'Test Min Recall': [min_recall],
                       'Test TNR': [train_tnr], 
                       'Test TNR Gap': [gap_train_tnr], 
                       'Test Min TNR': [min_tnr]}, 
                       index=[0])

        model_results_df = pd.concat([model_results_df, new_row], ignore_index=True)

    return model_results_df

### PLOTTING FUNCTIONS ###

def visualise_results(test_df,attribute,col_name,filter_group_size=True,min_size = 0.01):
    '''
    Make 4 plots with accuracy, precision/recall, sample size, and prevalence across subgroups (for one set of data ie train test or val!)
    args:
    test_df: results df (for train, val, or test data)
    attribute: name of sensitive attribute
    col_name: name of the column in df of sensitive attribute
    filter_group_size: whether to filter out subgroups with less than min_size
    min_size: minimum size of subgroup (proportion)
    
    returns:
    4 plots
    '''
    df = test_df
    accuracy, precision, recall, n_groups, pos_labels,tnr = get_stats(df,col_name,filter_group_size=filter_group_size,min_size=min_size)

    plt.figure(figsize=(10, 10))

    plt.subplot(2,2,1)
    plt.bar(accuracy.index, accuracy) # yerr=std_dev_ethnicity
    plt.title('Accuracy by ' + str(attribute))
    plt.ylim(0.5,1.0)
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 2)
    for i in range(len(recall)):
        plt.scatter(recall[i],precision[i],label=recall.index[i])
    plt.legend()
    plt.xlabel('Recall (TP/(TP+FN))')
    plt.ylabel('Precision (TP/(TP+FP))') # yerr=std_dev_ethnicity
    plt.ylim(0.5,1.0)
    plt.title('Precision/recall by ' + str(attribute))

    plt.subplot(2, 2, 3)
    plt.bar(n_groups.index, n_groups, capsize=5)
    plt.title('N per group')
    plt.ylim(0.5,1.0)
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 4)
    plt.bar(pos_labels.index, pos_labels, capsize=5)
    plt.title('Proportion of positive labels (ie = finding) by ' + str(attribute))
    plt.ylim(0.5,1.0)
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

def plot_results(index,test_df,data_name,attribute,col_name,shift=0,filter_group_size=True,min_size = 300):    
    '''
    function to get results for train val and test to be able to plot them together
    '''
    index=index # to keep track of which dataset is being plotted
    df = test_df
    accuracy, precision, recall, n_groups, pos_labels,tnr = get_stats(df,col_name,filter_group_size=filter_group_size,min_size=min_size)

    x = np.arange(len(accuracy))  # the label locations
    bar_width = 0.9  # Adjust the width as needed
    x_tick_labels=accuracy.index.tolist() # keep track of x tick labels (indices of each group for each df)
    
    plt.subplot(2,2,1)
    plt.bar(x+shift, accuracy,width=bar_width, align='edge') # need to adjust how much bars are shifted by based on how many groups
    plt.title('Accuracy by ' + str(attribute))
    plt.ylim(0.5,1.0)
    plt.legend(['Train','Val','Test'])

    plt.subplot(2, 2, 2)
    colors= ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o','x','v','s', '^','+','d']
    for i in range(len(recall)):
        plt.scatter(recall[i],precision[i],label=str(recall.index[i])+' ' + data_name,c=colors[index],marker = markers[i])
    plt.legend(fontsize="8")
    plt.xlabel('Recall (TP/(TP+FN))')
    plt.ylabel('Precision (TP/(TP+FP))') # yerr=std_dev_ethnicity
    plt.title('Precision/recall by ' + str(attribute))

    plt.subplot(2, 2, 3)
    plt.bar(x+shift, n_groups, width=bar_width, align='edge')
    plt.title('N per group')
    plt.legend(['Train','Val','Test'])

    plt.subplot(2, 2, 4)
    plt.bar(x+shift, pos_labels, width=bar_width, align='edge')
    plt.title('Proportion of positive labels (ie = finding) by ' + str(attribute))
    plt.legend(['Train','Val','Test'])

    return x_tick_labels

def visualise_all_results(list_of_dfs,data_names,attribute,col_name,filter_group_size=True,min_size = 0.05):
    '''
    Plot results for all datasets (train, val, test) on the same plot (accuracy, precision/recall, sample size, prevalence)
    '''
    plt.figure(figsize=(10, 10))

    shift=0
    x_tick_labels=[]
    for i in range(len(list_of_dfs)):
        x_tick_labels+= plot_results(i,list_of_dfs[i],data_names[i],attribute,col_name,shift,filter_group_size=filter_group_size,min_size=min_size)
        shift=len(x_tick_labels) # shift by how many groups have been plotted so far
    
    x_ticks = np.arange(0.5,len(x_tick_labels)+0.5)
    
    plt.subplot(2,2,1)
    plt.xticks(x_ticks,x_tick_labels,rotation=90)
    
    plt.subplot(2,2,3)
    plt.xticks(x_ticks,x_tick_labels,rotation=90)
    
    plt.subplot(2,2,4)
    plt.xticks(x_ticks,x_tick_labels,rotation=90)
    
    plt.tight_layout()
    plt.show()

def make_subplots(ax, metric_list, overall_metric_list, title,plot_gap,plot_all):
    if plot_gap:
        ax.plot([x.max()-x.min() for x in metric_list])
        ax.set_ylabel(title + ' gap')

    elif plot_all:
        ax.plot([x for x in metric_list])
        ax.set_ylabel(title)
        ax.set_ylim(0.0,1.02)
        ax.legend(range(len(metric_list)))
        #ax.legend(['a','b','c','d','e','f'])

    else:
        ax.plot([x.min() for x in metric_list])
        ax.plot([x for x in overall_metric_list]) ### THIS IS THE WRONG MEAN - NOT OVERALL MEAN JUST SUBGROUP MEAN
        ax.plot([x.max() for x in metric_list])
        ax.legend(['min','mean','max'])
        ax.set_ylabel(title)
        ax.set_ylim(0.0,1.0)


    ax.set_xlabel('Epoch')


def plot_subgroup_stats(train_preds, val_preds, col_name,plot_gap=False,plot_auc=False,plot_all=False):
    train_acc_list, train_precision_list, train_recall_list, train_f1_list,train_tnr_list = get_subgroup_stats(train_preds, col_name)
    val_acc_list, val_precision_list, val_recall_list, val_f1_list,val_tnr_list = get_subgroup_stats(val_preds, col_name)
    overall_train_acc,overall_val_acc,overall_train_precision,overall_val_precision,overall_train_recall,overall_val_recall,overall_train_tnr,overall_val_tnr = get_overall_metrics(train_preds,val_preds)

    fig, axs = plt.subplots(6, 2, sharey='row', figsize=(20, 12))

    make_subplots(axs[0, 0], train_acc_list, overall_train_acc, 'Training Accuracy',plot_gap=plot_gap,plot_all=plot_all)
    make_subplots(axs[0, 1], val_acc_list, overall_val_acc, 'Validation Accuracy',plot_gap=plot_gap,plot_all=plot_all)
    
    make_subplots(axs[1, 0], train_precision_list, overall_train_precision, 'Training Precision',plot_gap=plot_gap,plot_all=plot_all)
    make_subplots(axs[1, 1], val_precision_list, overall_val_precision,'Validation Precision',plot_gap=plot_gap,plot_all=plot_all)

    make_subplots(axs[2, 0], train_recall_list, overall_train_recall, 'Training Recall',plot_gap=plot_gap,plot_all=plot_all)
    make_subplots(axs[2, 1], val_recall_list, overall_val_recall, 'Validation Recall', plot_gap=plot_gap,plot_all=plot_all)
    
    if plot_auc:
        train_auc_list = get_subgroup_auc(train_preds,col_name)
        val_auc_list = get_subgroup_auc(val_preds,col_name)
        overall_train_auc,overall_val_auc = get_overall_auc(train_preds,val_preds)
       
        make_subplots(axs[3, 0], train_auc_list, overall_train_auc, 'Training AUC',plot_gap=plot_gap,plot_all=plot_all)
        make_subplots(axs[3, 1], val_auc_list, overall_val_auc,'Validation AUC',plot_gap=plot_gap,plot_all=plot_all)

    make_subplots(axs[4, 0], train_tnr_list, overall_train_tnr, 'Training TNR',plot_gap=plot_gap,plot_all=plot_all)
    make_subplots(axs[4, 1], val_tnr_list, overall_val_tnr, 'Validation TNR', plot_gap=plot_gap,plot_all=plot_all)

    train_balanced_acc_list = ((np.array(train_recall_list) + np.array(train_tnr_list))/2).tolist()
    val_balanced_acc_list = ((np.array(val_recall_list) + np.array(val_tnr_list))/2).tolist()

    make_subplots(axs[5, 0], train_balanced_acc_list, overall_train_tnr, 'Training Balanced Accuracy',plot_gap=plot_gap,plot_all=plot_all)
    make_subplots(axs[5, 1], val_balanced_acc_list, overall_val_tnr, 'Validation Balanced Accuracy', plot_gap=plot_gap,plot_all=plot_all)


    #plt.suptitle('Fairness during training for ' + col_name + ' subgroups')
    plt.tight_layout()
    plt.show()

def plot_subgroup_model_stats(train_preds_list, val_preds_list, col_name,plot_gap=False,plot_auc=False,plot_all=False):
    n_models = len(train_preds_list)
    fig, axs = plt.subplots(n_models, 2, sharey='row', figsize=(20, 12))

    for i,train_preds in enumerate(train_preds_list):
        val_preds = val_preds_list[i]
        train_acc_list, train_precision_list, train_recall_list, train_f1_list,train_tnr_list = get_subgroup_stats(train_preds, col_name)
        val_acc_list, val_precision_list, val_recall_list, val_f1_list,val_tnr_list = get_subgroup_stats(val_preds, col_name)
        overall_train_acc,overall_val_acc,overall_train_precision,overall_val_precision,overall_train_recall,overall_val_recall,overall_train_tnr,overall_val_tnr = get_overall_metrics(train_preds,val_preds)
        train_auc_list = get_subgroup_auc(train_preds,col_name)
        val_auc_list = get_subgroup_auc(val_preds,col_name)
        overall_train_auc,overall_val_auc = get_overall_auc(train_preds,val_preds)

        make_subplots(axs[i, 0], train_auc_list, overall_train_auc, 'Training AUC',plot_gap=plot_gap,plot_all=plot_all)
        make_subplots(axs[i, 1], val_auc_list, overall_val_auc,'Validation AUC',plot_gap=plot_gap,plot_all=plot_all)

        # make_subplots(axs[0, 0], train_acc_list, overall_train_acc, 'Training Accuracy',plot_gap=plot_gap,plot_all=plot_all)
        # make_subplots(axs[0, 1], val_acc_list, overall_val_acc, 'Validation Accuracy',plot_gap=plot_gap,plot_all=plot_all)
        
        # make_subplots(axs[1, 0], train_precision_list, overall_train_precision, 'Training Precision',plot_gap=plot_gap,plot_all=plot_all)
        # make_subplots(axs[1, 1], val_precision_list, overall_val_precision,'Validation Precision',plot_gap=plot_gap,plot_all=plot_all)

        # make_subplots(axs[2, 0], train_recall_list, overall_train_recall, 'Training Recall',plot_gap=plot_gap,plot_all=plot_all)
        # make_subplots(axs[2, 1], val_recall_list, overall_val_recall, 'Validation Recall', plot_gap=plot_gap,plot_all=plot_all)

        # make_subplots(axs[i, 0], train_tnr_list, overall_train_tnr, 'Training TNR',plot_gap=plot_gap,plot_all=plot_all)
        # make_subplots(axs[i, 1], val_tnr_list, overall_val_tnr, 'Validation TNR', plot_gap=plot_gap,plot_all=plot_all)

        # train_balanced_acc_list = ((np.array(train_recall_list) + np.array(train_tnr_list))/2).tolist()
        # val_balanced_acc_list = ((np.array(val_recall_list) + np.array(val_tnr_list))/2).tolist()

        # make_subplots(axs[5, 0], train_balanced_acc_list, overall_train_tnr, 'Training Balanced Accuracy',plot_gap=plot_gap,plot_all=plot_all)
        # make_subplots(axs[5, 1], val_balanced_acc_list, overall_val_tnr, 'Validation Balanced Accuracy', plot_gap=plot_gap,plot_all=plot_all)

        axs[i, 0].set_ylim([0.4,0.8])
        axs[i, 1].set_ylim([0.4,0.8])

    #plt.suptitle('Fairness during training for ' + col_name + ' subgroups')
    plt.tight_layout()
    plt.show()

def conduct_pca(features,test_preds,n_pc=5):
    '''
    Function to conduct PCA on the data and get results_df with first n PC's and different attributes (with standard scaling)
    args:
    features: torch.tensor where each row corresponds to an image and each col a different feature
    test_preds: dataframe with the predictions and metadata
    n_pc: number of principal components (actually for now i don't support =/= 5)
    returns:
    results_df: dataframe with the first n principal components and different attributes
    (also prints pca explained variance)
    '''
    X = features.numpy()
    std_scaler = StandardScaler()
    scaled_X = std_scaler.fit_transform(X)
    pca = PCA(n_components=n_pc)
    pconp = pca.fit_transform(scaled_X)
    print(pca.explained_variance_ratio_)
    #components = pd.DataFrame(components).transpose()

    # could write this bit better!
    results_df = pd.DataFrame(pconp,columns=['PC0','PC1','PC2','PC3','PC4'])
    Y = test_preds['binaryLabel'].values
    results_df['label']=Y
    results_df['label'] = results_df['label'].astype('category')
    results_df['pred'] = test_preds['pred'].values
    results_df['pred'] = results_df['pred'].astype('category')
    results_df['raw_pred_cat'] = test_preds['raw_pred'].values
    #results_df['raw_pred_cat'] = test_preds['raw_pred'].astype('category')
    results_df['raw_pred_cat'] = pd.cut(results_df['raw_pred_cat'],bins=[0,0.2,0.4,0.6,0.8,1],labels=['0','1','2','3','4'])
    results_df['sex'] = test_preds['Sex'].values
    results_df['bmi_cat'] = test_preds['bmi_cat'].values
    results_df['bmi_cat'] = results_df['bmi_cat'].astype('category')
    results_df['ethnicity'] = test_preds['ethnicity'].values
    results_df['ethnicity'] = results_df['ethnicity'].astype('category')
    results_df['Age_multi'] = test_preds['Age_multi'].values
    results_df['Age_multi'] = results_df['Age_multi'].astype('category')
    results_df['alcohol'] = test_preds['alcohol'].values
    results_df['alcohol'] = results_df['alcohol'].astype('category')

    results_df['assessment_centre'] = test_preds['assessment_centre'].values
    results_df['assessment_centre'] = results_df['assessment_centre'].astype('category')

    return results_df

def conduct_tsne(features, test_preds, n_components=2):
    '''
    Function to conduct t-SNE on the data and get results_df with first n components and different attributes (with standard scaling)
    args:
    features: torch.tensor where each row corresponds to an image and each col a different feature
    test_preds: dataframe with the predictions and metadata
    n_components: number of components (default is 2 for 2D visualization)
    returns:
    results_df: dataframe with the first n components and different attributes
    '''
    X = features.numpy()
    std_scaler = StandardScaler()
    scaled_X = std_scaler.fit_transform(X)
    tsne = TSNE(n_components=n_components)
    tsne_results = tsne.fit_transform(scaled_X)

    # Create a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE0', 'TSNE1'])

    # Add the other attributes from test_preds
    for col in ['binaryLabel', 'pred', 'raw_pred', 'Sex', 'bmi_cat', 'ethnicity', 'Age_multi', 'alcohol', 'assessment_centre']:
        tsne_df[col] = test_preds[col].values
        tsne_df[col] = tsne_df[col].astype('category')

    tsne_df['raw_pred_cat'] = test_preds['raw_pred'].values
    tsne_df['raw_pred_cat'] = pd.cut(tsne_df['raw_pred_cat'],bins=[0,0.2,0.4,0.6,0.8,1],labels=['0','1','2','3','4'])
    return tsne_df

## CONDITION ON CERTAIN GROUPS ##

def conditional_metrics(test_preds,grouped_cols,one_class=False):
    # grouped_cols can be one or multiple col names
    grouped_df = test_preds.groupby(grouped_cols)
    accuracy = grouped_df.apply(lambda group: (group['binary_label'] == group['pred']).mean())
    precision = grouped_df.apply(lambda group: (group['TP']).sum()/((group['TP']).sum()+(group['FP']).sum()))
    recall = grouped_df.apply(lambda group: (group['TP']).sum()/((group['FN']).sum()+(group['TP']).sum()))
    if one_class:
        results_df = pd.DataFrame({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
        })
    else:
        auc = grouped_df.apply(lambda group:roc_auc_score(group['binary_label'], group['raw_pred']))
        tnr = grouped_df.apply(lambda group: ((group['TN']).sum())/((group['TN']).sum()+(group['FP']).sum()))
        results_df = pd.DataFrame({
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'TNR': tnr
        })
    # to get average across multiple test_preds: df_mean = pd.concat(list_of_results_df).groupby(level=0).mean(), but not quite surw how to o this when there are two groupby cols
    return results_df.round(3)

def calculate_metrics(df, pred_col, label_col='binaryLabel'):
    auc_col = 'raw_pred'
    auc = roc_auc_score(df[label_col], df[auc_col])
    accuracy = (df[pred_col] == df[label_col]).mean()
    balanced_accuracy = balanced_accuracy_score(df[pred_col], df[label_col])
    precision = ((df[pred_col] == 1) & (df[pred_col] == df[label_col])).sum() / df[pred_col].sum()
    recall = ((df[pred_col] == 1) & (df[pred_col] == df[label_col])).sum() / df[label_col].sum()
    tnr = ((df[pred_col] == 0) & (df[pred_col] == df[label_col])).sum() / (df[label_col] == 0).sum()
    return {'AUC': auc, 'Accuracy': accuracy, 'Balanced Accuracy': balanced_accuracy, 'Precision': precision, 'Recall': recall, 'TNR': tnr}