#****************************************
# MIT License
# Copyright (c) 2025 Zexu Li, Jinying Chen
#  
# author(s): Zexu Li, Jinying Chen, Boston University Chobanian & Avedisian School of Medicine
# date: 2025-7-7
# ver: 1.0
# 
# This code was written to support data analysis for the Data Harmonization Using Natural Language 
# Processing (NLP harmonization) project and the 2025 paper published in PLOS One.
# The code is for research use only, and is provided as it is.
# 

# Python code that runs the 50 trials for the ML experiments in High Performance Computing Cluster (HPCC)
# output from each trial will be automatically stored in output1.txt, output2.txt, ..., during batch processing in HPCC

import pandas as pd
from sklearn.model_selection import train_test_split
Final_df_ML = pd.read_csv('ML_dataset_021825_v3.csv')

import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score,f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import os


def extract_train_sources(text):
    # Use regex to find the 'Train Source' and 'Test Source' parts
    train_match = re.search(r"Train Source:\[(.*?)\], Test Source", text)
    
    if train_match:
        train_sources = train_match.group(1)
        # Split the sources by separating them with ' ' and remove any empty strings
        train_sources_list = [source for source in train_sources.split("'") if source.strip()]
        return train_sources_list
    else:
        return ""
    
def extract_between_phrases(file_path, start_phrase, end_phrase):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    start_index = -1
    end_index = -1
    for i, line in enumerate(lines):
        if line.startswith(start_phrase):
            start_index = i
        if end_phrase in line:
            end_index = i
            break
    
    if start_index != -1 and end_index != -1:
        return ''.join([line.strip() for line in lines[start_index:end_index+1]])
    else:
        return []


def select_true_and_false(df_source):
    true_values = df_source[df_source['Mapping_result'] == 1]
    false_values = df_source[df_source['Mapping_result'] == 0].sample(n=200*len(true_values), random_state=42)
    return pd.concat([true_values, false_values])

def split_into_sublists(lst, k):
    """
    Split a list into k sublists without overlapping.

    Parameters:
    - lst: List to be split.
    - k: Number of sublists.

    Returns:
    A list of k sublists.
    """
    n = len(lst)
    sublist_size = n // k
    remainder = n % k

    sublists = []
    start = 0

    for i in range(k):
        sublist_length = sublist_size + (1 if i < remainder else 0)
        sublists.append(lst[start:start + sublist_length])
        start += sublist_length

    return sublists

id = os.getenv('SGE_TASK_ID')

# randomly split source variables into train and test sets
# method 1: when running the ML expts for the first time in High Performance Computing Cluster
'''
# Identify unique values in the 'source' column
unique_sources = Final_df_ML['Source'].unique()
# Choose a subset of unique sources for training and testing
train_sources, test_sources = train_test_split(unique_sources, test_size=0.2, random_state = 42+id)
print(f'Train Source:{train_sources}, Test Source:{test_sources}')
'''
# end of method 1

# method 2: when repeating the ML expts using train:test splits for the 50 trials done before
# extracting train:test split of the source variables from the output files (i.e., output1.txt, output2.txt, ...) 
# from a previous ML experiment
file_path = 'output' + str(id) + '.txt'
text = extract_between_phrases(file_path,'Train Source','Test Source')
train_sources = extract_train_sources(text)
train = Final_df_ML[Final_df_ML['Source'].isin(train_sources)]
test = Final_df_ML[~Final_df_ML['Source'].isin(train_sources)]
unique_test_sources = test['Source'].unique()
print(f'Train Source:{train_sources}, Test Source:{unique_test_sources}')
# end of method 2

split_train_source = split_into_sublists(train_sources, 5)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score,f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
param_grid = {
        'n_estimators': [50, 200,500,700,900,1000],
        'max_depth': [2, 10,15,20],
        'criterion': ['gini', 'entropy'],
        'max_features':['sqrt', None]
        
    }
best_HR = -1
best_MRR = -1
best_unique_HR = -1
n = 0
mean_HR_list = []
mean_MRR_list = []
mean_unique_HR_list = []
para_list = []
for g in ParameterGrid(param_grid):
        #rf = RandomForestClassifier(random_state=42)
        #rf.set_params(**g)
        HR_list = []
        unique_HR_list = []
        MRR_list = []
        for split in split_train_source:
            rf = RandomForestClassifier(random_state=42)
            rf.set_params(**g)
            validation_data = train[train['Source'].isin(split)]
            train_data = train[~train['Source'].isin(split)]
            
            final_train_set = train_data.groupby('Source').apply(select_true_and_false).reset_index(drop=True)
            final_validation_set = validation_data.copy()
            

            train_dataonly = final_train_set[['miniLM_on_label', 'e5_on_label', 'mpnet_on_label', 'fuzzy_on_label', 'biolord_on_label',
                                'miniLM_on_label_key', 'e5_on_label_key', 'mpnet_on_label_key', 'fuzzy_on_label_key', 'biolord_on_label_key',
                                'miniLM_on_sheet', 'e5_on_sheet', 'mpnet_on_sheet', 'fuzzy_on_sheet', 'biolord_on_sheet',
                               'deriv_info_null_EU','deriv_info_len_EU','Label_len_EU',
                               'deriv_info_null_JP','deriv_info_len_JP','Label_len_JP','Mapping_result']]
            validation_dataonly = final_validation_set[['miniLM_on_label', 'e5_on_label', 'mpnet_on_label', 'fuzzy_on_label','biolord_on_label',
                                'miniLM_on_label_key', 'e5_on_label_key', 'mpnet_on_label_key', 'fuzzy_on_label_key','biolord_on_label_key',
                                'miniLM_on_sheet', 'e5_on_sheet', 'mpnet_on_sheet', 'fuzzy_on_sheet','biolord_on_sheet',
                               'deriv_info_null_EU','deriv_info_len_EU','Label_len_EU',
                               'deriv_info_null_JP','deriv_info_len_JP','Label_len_JP','Mapping_result']]

            X_train = train_dataonly.drop('Mapping_result',axis = 1)
            y_train = train_dataonly[['Mapping_result']].values.ravel()
            X_validation = validation_dataonly.drop('Mapping_result',axis = 1)
            y_validation = validation_dataonly[['Mapping_result']].values.ravel()
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_validation)
            y_pred_proba = rf.predict_proba(X_validation)[:, 1]
            final_validation_set['probability'] = y_pred_proba
            final_validation_set['rank'] = final_validation_set.groupby('Source')['probability'].rank(ascending=False)
            positive = final_validation_set[final_validation_set['Mapping_result'] == 1]
            
            unique_positive = positive.groupby('Source')['rank'].min().reset_index()
            top_30_HR_unique = len(unique_positive[unique_positive['rank'] <=30])/len(unique_positive)
            
            top30_HR = len(positive[positive['rank'] <=30])/ len(positive)
            HR_list.append(top30_HR)
            unique_HR_list.append(top_30_HR_unique)
            max_ranks = positive.groupby('Source')['rank'].idxmin()
            result_df = positive.loc[max_ranks]
            result_df['MRR'] = 1/result_df['rank']


            MRR = sum(result_df['MRR'])/ len(result_df)
            MRR_list.append(MRR)
        mean_HR = sum(HR_list) / len(HR_list)
        mean_unique_HR = sum(unique_HR_list)/len(unique_HR_list)
        mean_MRR = sum(MRR_list) / len(MRR_list)
        if mean_HR > best_HR:
            best_HR = mean_HR
            best_grid_HR = g
        if mean_MRR > best_MRR:
            best_MRR = mean_MRR
            best_grid_MRR = g
        if mean_unique_HR > best_unique_HR:
            best_unique_HR = mean_unique_HR
            best_grid_unique_HR = g
        
        n+=1
        mean_HR_list.append(mean_HR)   
        mean_MRR_list.append(mean_MRR)
        mean_unique_HR_list.append(mean_unique_HR)
        para_list.append(g)
        print(f'Step {n} complete: model:{rf}, HR: {mean_HR}, unique_HR:{mean_unique_HR}, MRR: {mean_MRR}')


for grid in [best_grid_HR, best_grid_MRR, best_grid_unique_HR]:
    rf = RandomForestClassifier(random_state=42)
    rf.set_params(**grid)
    train = Final_df_ML[Final_df_ML['Source'].isin(train_sources)]
    test = Final_df_ML[~Final_df_ML['Source'].isin(train_sources)]


    test_copy = test.copy()

    test_data = test[['miniLM_on_label', 'e5_on_label', 'mpnet_on_label', 'fuzzy_on_label','biolord_on_label',
                                    'miniLM_on_label_key', 'e5_on_label_key', 'mpnet_on_label_key', 'fuzzy_on_label_key','biolord_on_label_key',
                                    'miniLM_on_sheet', 'e5_on_sheet', 'mpnet_on_sheet', 'fuzzy_on_sheet','biolord_on_sheet',
                                   'deriv_info_null_EU','deriv_info_len_EU','Label_len_EU',
                                   'deriv_info_null_JP','deriv_info_len_JP','Label_len_JP','Mapping_result']]
    final_train_set = train.groupby('Source').apply(select_true_and_false).reset_index(drop=True)

    train_dataonly = final_train_set[['miniLM_on_label', 'e5_on_label', 'mpnet_on_label', 'fuzzy_on_label','biolord_on_label',
                                    'miniLM_on_label_key', 'e5_on_label_key', 'mpnet_on_label_key', 'fuzzy_on_label_key','biolord_on_label_key',
                                    'miniLM_on_sheet', 'e5_on_sheet', 'mpnet_on_sheet', 'fuzzy_on_sheet','biolord_on_sheet',
                                   'deriv_info_null_EU','deriv_info_len_EU','Label_len_EU',
                                   'deriv_info_null_JP','deriv_info_len_JP','Label_len_JP','Mapping_result']]

        

    X_test = test_data.drop('Mapping_result',axis = 1)
    y_test = test_data[['Mapping_result']].values.ravel()

    X_train = train_dataonly.drop('Mapping_result',axis = 1)
    y_train = train_dataonly[['Mapping_result']].values.ravel()


    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    test_copy['probability'] = y_pred_proba
    test_copy['rank'] = test_copy.groupby('Source')['probability'].rank(ascending=False)
    test_copy['rank_e5'] = test_copy.groupby('Source')['e5_on_label'].rank(ascending=False)
    
    positive = test_copy[test_copy['Mapping_result'] == 1]

    unique_positive = positive.groupby('Source')['rank'].min().reset_index()
    unique_positive_e5 = positive.groupby('Source')['rank_e5'].min().reset_index()
    
    top_30_HR_unique_e5 = len(unique_positive[unique_positive_e5['rank_e5'] <=30])/len(unique_positive)
    top_20_HR_unique_e5 = len(unique_positive[unique_positive_e5['rank_e5'] <=20])/len(unique_positive)
    top_10_HR_unique_e5 = len(unique_positive[unique_positive_e5['rank_e5'] <=10])/len(unique_positive)
    top_5_HR_unique_e5 = len(unique_positive[unique_positive_e5['rank_e5'] <=5])/len(unique_positive)
    
    top_30_HR_unique = len(unique_positive[unique_positive['rank'] <=30])/len(unique_positive)
    top_20_HR_unique = len(unique_positive[unique_positive['rank'] <=20])/len(unique_positive)
    top_10_HR_unique = len(unique_positive[unique_positive['rank'] <=10])/len(unique_positive)
    top_5_HR_unique = len(unique_positive[unique_positive['rank'] <=5])/len(unique_positive)

    #top30_HR = len(positive[positive['rank'] <=30])/ len(positive)
    #top20_HR = len(positive[positive['rank'] <=20])/ len(positive)
    #top10_HR = len(positive[positive['rank'] <=10])/ len(positive)
    #top5_HR = len(positive[positive['rank'] <=5])/ len(positive)

    max_ranks = positive.groupby('Source')['rank'].idxmin()
    max_ranks_e5 = positive.groupby('Source')['rank_e5'].idxmin()
    result_df = positive.loc[max_ranks]
    result_df_e5 = positive.loc[max_ranks_e5]
    
    result_df['MRR'] = 1/result_df['rank']
    MRR = sum(result_df['MRR'])/ len(result_df)
    
    result_df_e5['MRR'] = 1/result_df_e5['rank_e5']
    MRR_e5 = sum(result_df_e5['MRR'])/ len(result_df_e5)
    
    print(f'Best Grid:{grid}, unique_HR:{top_30_HR_unique},{top_20_HR_unique},{top_10_HR_unique},{top_5_HR_unique}, MRR:{MRR}')
    print(f'New E5, unique_HR:{top_30_HR_unique_e5},{top_20_HR_unique_e5},{top_10_HR_unique_e5},{top_5_HR_unique_e5}, MRR:{MRR_e5}')


