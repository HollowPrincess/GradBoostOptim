import pandas as pd
import numpy as np
import math
import os
from time import time
import json 

from sklearn.utils import shuffle

import xgboost

from scipy.stats import randint as sp_randint # int distribution for random search
from scipy.stats import uniform # float distribution for random search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from matplotlib import pyplot as plt

import os

def group_best_params_with_stop(df, stop_iter_num, left_iter_num, right_iter_num):
    best_params = {}
    stop_iter_nums = {}
    best_score = {}

    tmp_df = df.loc[(df['iter_num'] >= left_iter_num) & (df['iter_num'] <= right_iter_num)].copy() #take group
    for i in tmp_df.run_number.unique(): 
        tmp_df_i = tmp_df.loc[tmp_df.run_number==i] #take run
        #cut iterations:
        tmp_df_i_copy = tmp_df_i.copy()
        tmp_df_i_copy.loc[:, "mean_test_score"] = tmp_df_i.loc[:, "mean_test_score"].cummax()      

        try:
            curr_max=tmp_df_i_copy.groupby("mean_test_score").count().reset_index().sort_values(by = "mean_test_score")
            max_score = curr_max.loc[curr_max["mean_test_score"]>=stop_iter_num].index[0] #get first interval with iter_amount more than stop_iter_num
        except:
            max_score = tmp_df_i_copy.mean_test_score.max()

        tmp_stop = tmp_df_i.loc[tmp_df_i["mean_test_score"]==max_score].sort_values(by="iter_num").iloc[0]#get first element with best_score

        best_params[i] = tmp_stop['params']
        #best_score[i] = tmp_stop['mean_test_score']

        last_iter_num = tmp_stop['iter_num'] + stop_iter_num - 1
        if last_iter_num > right_iter_num:
            last_iter_num = right_iter_num

        stop_iter_nums[i] = last_iter_num
    return best_params, stop_iter_nums#, best_score   

def run():
    PREP_DATA_PATH = "./data/input/prepared_facebook_data.csv"
    data=pd.read_csv(PREP_DATA_PATH)

    RES_DATA_DIR = "./data/results/test 1"
    RES_SAVE_DIR = "./data/results/test 2"

    files = os.listdir(RES_DATA_DIR)
    files = [x for x in files if x.find("group_search")!=-1]

        

    group_df = pd.DataFrame([])
    for file_res in files:
        df = pd.read_csv(RES_DATA_DIR+"/"+file_res)    
        df['iter_num'] = range(1, df.shape[0]+1)
        group_df = pd.concat([group_df, df], sort = False)

    # get some data
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values

    param_dist_group_2 = {"alpha": uniform(loc=0, scale=1),
                      "lambda": uniform(loc=0, scale=1)}

    param_dist_group_3 = {"subsample":uniform(loc=0.5, scale=0.4),
                      "colsample_bytree":uniform(loc=0.5, scale=0.4)}


    stop_iter_num = 50

    best_params_first_group, first_stop_iter_nums = group_best_params_with_stop(group_df, 
                                                                                stop_iter_num = stop_iter_num,
                                                                                left_iter_num=1, 
                                                                                right_iter_num=243)


    #for params_set_num in best_params_first_group:

    for params_set_num in [2,4,6]:
        #clf = xgboost.XGBRegressor(tree_method = "gpu_hist", gpu_id=0, verbosity=0)
        clf = xgboost.XGBRegressor(verbosity=0)
        clf.set_params(**json.loads(best_params_first_group[params_set_num].replace("'", "\"")))
        n_iter_search_sec = int(round((729 - first_stop_iter_nums[params_set_num])/2))
        #second group:
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist_group_2,
                                          n_iter=n_iter_search_sec, cv=5, scoring="r2")

        random_search.fit(X, y)

        res_rand_2=pd.DataFrame(random_search.cv_results_)
        res_rand_2['iter_num'] = range(first_stop_iter_nums[params_set_num]+1, first_stop_iter_nums[params_set_num]+n_iter_search_sec+1)
        res_rand_2['experiment_name']='random search for 2 group' 
        res_rand_2['run_number']=params_set_num

        #fit last suitable params
        best_params_second_group, sec_stop_iter_nums = group_best_params_with_stop(res_rand_2, 
                                                             stop_iter_num = stop_iter_num, 
                                                             left_iter_num=first_stop_iter_nums[params_set_num]+1, 
                                                             right_iter_num=n_iter_search_sec)
        clf.set_params(**best_params_second_group[params_set_num])
        print('done for second group')

        #third group:
        n_iter_search_third = 729 - sec_stop_iter_nums[params_set_num]
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist_group_3,
                                            n_iter=n_iter_search_third, cv=5, scoring="r2")
        random_search.fit(X, y)

        res_rand_3=pd.DataFrame(random_search.cv_results_)
        res_rand_3['iter_num'] = range(sec_stop_iter_nums[params_set_num]+1, 729+1)
        res_rand_3['experiment_name']='random search for 3 group' 
        res_rand_3['run_number']=params_set_num

        best_params_third_group, third_stop_iter_nums = group_best_params_with_stop(res_rand_3, 
                                                             stop_iter_num = stop_iter_num, 
                                                             left_iter_num=sec_stop_iter_nums[params_set_num]+1, 
                                                             right_iter_num=729)
        res_rand_3=res_rand_3.loc[(res_rand_3['run_number']==params_set_num) & \
                              (res_rand_3['iter_num']<=int(third_stop_iter_nums[params_set_num]))]

        res=pd.concat([res_rand_2,res_rand_3], sort=False)

        sec_idxs = res.loc[res['experiment_name']=='random search for 2 group','iter_num'].values
        third_idxs = res.loc[res['experiment_name']=='random search for 3 group','iter_num'].values
        intersection = np.intersect1d(third_idxs, sec_idxs)

        res = res.loc[~((res['experiment_name']=='random search for 2 group') & (res['iter_num'].isin(intersection)))]

        res.to_csv(RES_SAVE_DIR+'/GR_S_2_and_3_groups_with_'+str(stop_iter_num)+'_'+str(params_set_num), index=False)
        print('done for third group')



