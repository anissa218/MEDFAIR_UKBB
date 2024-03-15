import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import roc_auc_score
import warnings

### PRE PROCESSING FUNCTIONS ###

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
        path_to_preds = root_path + file_name
        path_to_splits = path_to_val_splits
        val_preds[epoch_num]= preprocessing_function(path_to_preds,path_to_splits) # replace the file name with the processed dataframe

    for epoch_num, file_name in train_preds.items():
        path_to_preds = root_path + file_name
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
    for epoch_num in range(len(val_preds)):
        val_df = val_preds[epoch_num]
        train_df = train_preds[epoch_num]
        val_acc.append(len(val_df[val_df['pred']==val_df['binary_label']])/len(val_df))
        train_acc.append(len(train_df[train_df['pred']==train_df['binary_label']])/len(train_df))
        val_precision.append(len(val_df[val_df['TP']==1])/(len(val_df[val_df['TP']==1])+len(val_df[val_df['FP']==1])))
        train_precision.append(len(train_df[train_df['TP']==1])/(len(train_df[train_df['TP']==1])+len(train_df[train_df['FP']==1])))
        val_recall.append(len(val_df[val_df['TP']==1])/(len(val_df[val_df['TP']==1])+len(val_df[val_df['FN']==1])))
        train_recall.append(len(train_df[train_df['TP']==1])/(len(train_df[train_df['TP']==1])+len(train_df[train_df['FN']==1])))
    return train_acc,val_acc,train_precision,val_precision,train_recall,val_recall

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

    
    return accuracy, precision, recall, n_groups, pos_labels


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

def get_subgroup_stats(preds,col_name):
    '''
    Function to analyse same metrics but for each subgroup
    '''
    train_preds = preds
    col_name = col_name
    train_acc_list,train_precision_list, train_recall_list,train_f1_list= [],[],[],[]

    for epoch_num in range(len(train_preds)): # should probably fine a more elegant way to do this
        # train data:
        accuracy, precision, recall, n_groups, pos_labels = get_stats(train_preds[epoch_num],col_name)
        f1 = 2*(precision*recall)/(precision+recall)

        train_acc_list.append(accuracy)
        train_precision_list.append(precision)
        train_recall_list.append(recall)
        train_f1_list.append(f1)
    
    return train_acc_list,train_precision_list, train_recall_list,train_f1_list


def make_results_df(train_preds,val_preds,subgroups):
    '''
    Function for quantitative comparison. Still need to add AUC
    Just looks at results at final epoch
    '''
    model_results_df = pd.DataFrame(columns=['Subgroup','Train Acc', 'Train Acc Gap','Val Acc','Val Acc Gap','Train Precision','Train Precision Gap','Val Precision','Val Precision Gap','Train Recall','Train Recall Gap','Val Recall','Val Recall Gap'])

    for subgroup in subgroups:
        train_acc_list,train_precision_list, train_recall_list,train_f1_list = get_subgroup_stats(train_preds,subgroup)
        val_acc_list, val_precision_list, val_recall_list,val_f1_list = get_subgroup_stats(val_preds,subgroup)
        # # AUC only works if you have enough examples of each class
        # train_auc = get_subgroup_auc(train_preds,subgroup)[-1]
        # val_auc = get_subgroup_auc(val_preds,subgroup)[-1]
        train_acc,train_precision, train_recall,train_f1 = train_acc_list[-1],train_precision_list[-1], train_recall_list[-1],train_f1_list[-1]
        val_acc,val_precision, val_recall,val_f1 = val_acc_list[-1], val_precision_list[-1], val_recall_list[-1],val_f1_list[-1]
        gap_train_acc,gap_train_precision,gap_train_recall = train_acc.max()-train_acc.min(),train_precision.max()-train_precision.min(),train_recall.max()-train_recall.min()
        gap_val_acc,gap_val_precision,gap_val_recall = val_acc.max()-val_acc.min(),val_precision.max()-val_precision.min(),val_recall.max()-val_recall.min()
        #gap_train_auc,gap_val_auc = train_auc.max()-train_auc.min(),val_auc.max()-val_auc.min()


        train_acc,val_acc,train_precision,val_precision,train_recall,val_recall=get_overall_metrics(train_preds,val_preds) # overall metrics, independent of subgroup
        train_acc,val_acc,train_precision,val_precision,train_recall,val_recall = train_acc[-1],val_acc[-1],train_precision[-1],val_precision[-1],train_recall[-1],val_recall[-1] # only get last epoch
        # train_auc,val_auc = get_overall_auc(train_preds,val_preds)
        # train_auc,val_auc = train_auc[-1],val_auc[-1]

        #model_results_df = model_results_df.append({'Subgroup':subgroup,'Train Acc':train_acc,'Train Acc Gap':gap_train_acc,'Val Acc':val_acc,'Val Acc Gap':gap_val_acc,'Train Precision':train_precision,'Train Precision Gap':gap_train_precision,'Val Precision':val_precision,'Val Precision Gap':gap_val_precision,'Train Recall':train_recall,'Train Recall Gap':gap_train_recall,'Val Recall':val_recall,'Val Recall Gap':gap_val_recall}, ignore_index = True) #'Train AUC':train_auc, 'Train AUC Gap': gap_train_auc, 'Val AUC': val_auc, 'Val AUC Gap': gap_val_auc},ignore_index=True)
    
        new_row = pd.DataFrame({'Subgroup': [subgroup],
                        'Train Acc': [train_acc],
                        'Train Acc Gap': [gap_train_acc],
                        'Val Acc': [val_acc],
                        'Val Acc Gap': [gap_val_acc],
                        'Train Precision': [train_precision],
                        'Train Precision Gap': [gap_train_precision],
                        'Val Precision': [val_precision],
                        'Val Precision Gap': [gap_val_precision],
                        'Train Recall': [train_recall],
                        'Train Recall Gap': [gap_train_recall],
                        'Val Recall': [val_recall],
                        'Val Recall Gap': [gap_val_recall]}, 
                       #'Train AUC': [train_auc], 
                       #'Train AUC Gap': [gap_train_auc], 
                       #'Val AUC': [val_auc], 
                       #'Val AUC Gap': [gap_val_auc]}, 
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
    accuracy, precision, recall, n_groups, pos_labels = get_stats(df,col_name,filter_group_size=filter_group_size,min_size=min_size)

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
    plt.title('Precision/recall by ' + str(attribute))

    plt.subplot(2, 2, 3)
    plt.bar(n_groups.index, n_groups, capsize=5)
    plt.title('N per group')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 4)
    plt.bar(pos_labels.index, pos_labels, capsize=5)
    plt.title('Proportion of positive labels (ie = finding) by ' + str(attribute))
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

def plot_results(index,test_df,data_name,attribute,col_name,shift=0,filter_group_size=True,min_size = 300):    
    '''
    function to get results for train val and test to be able to plot them together
    '''
    index=index # to keep track of which dataset is being plotted
    df = test_df
    accuracy, precision, recall, n_groups, pos_labels = get_stats(df,col_name,filter_group_size=filter_group_size,min_size=min_size)

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

def make_subplots(ax, metric_list, overall_metric_list, title,plot_gap):
    if plot_gap:
        ax.plot([x.max()-x.min() for x in metric_list])
        ax.set_ylabel(title + ' gap')

    else:
        ax.plot([x.min() for x in metric_list])
        ax.plot([x for x in overall_metric_list]) ### THIS IS THE WRONG MEAN - NOT OVERALL MEAN JUST SUBGROUP MEAN
        ax.plot([x.max() for x in metric_list])
        ax.legend(['min','mean','max'])
        ax.set_ylabel(title)


    ax.set_xlabel('Epoch')

def plot_subgroup_stats(train_preds, val_preds, col_name,plot_gap=False,plot_auc=False):
    train_acc_list, train_precision_list, train_recall_list, train_f1_list = get_subgroup_stats(train_preds, col_name)
    val_acc_list, val_precision_list, val_recall_list, val_f1_list = get_subgroup_stats(val_preds, col_name)
    overall_train_acc,overall_val_acc,overall_train_precision,overall_val_precision,overall_train_recall,overall_val_recall = get_overall_metrics(train_preds,val_preds)

    fig, axs = plt.subplots(4, 2, sharey='row', figsize=(18, 18))

    make_subplots(axs[0, 0], train_acc_list, overall_train_acc, 'Training Accuracy',plot_gap=plot_gap)
    make_subplots(axs[0, 1], val_acc_list, overall_val_acc, 'Validation Accuracy',plot_gap=plot_gap)
    
    make_subplots(axs[1, 0], train_precision_list, overall_train_precision, 'Training Precision',plot_gap=plot_gap)
    make_subplots(axs[1, 1], val_precision_list, overall_val_precision,'Validation Precision',plot_gap=plot_gap)

    make_subplots(axs[2, 0], train_recall_list, overall_train_recall, 'Training Recall',plot_gap=plot_gap)
    make_subplots(axs[2, 1], val_recall_list, overall_val_recall, 'Validation Recall', plot_gap=plot_gap)
    
    if plot_auc:
        train_auc_list = get_subgroup_auc(train_preds,col_name)
        val_auc_list = get_subgroup_auc(val_preds,col_name)
        overall_train_auc,overall_val_auc = get_overall_auc(train_preds,val_preds)
       
        make_subplots(axs[3, 0], train_auc_list, overall_train_auc, 'Training AUC',plot_gap=plot_gap)
        make_subplots(axs[3, 1], val_auc_list, overall_val_auc,'Validation AUC',plot_gap=plot_gap)


    plt.suptitle('Fairness during training for ' + col_name + ' subgroups')
    plt.tight_layout()
    plt.show()

