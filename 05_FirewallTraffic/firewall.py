import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import time
from itertools import combinations

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler

############################## Load Data ##############################

############################## End Load Data ##############################



############################## Initial Preprocessing ##############################

############################## End Initial Preprocessing ##############################



############################## EDA ##############################

############################## END EDA ##############################


############################## Baseline Models ##############################
def create_train_and_final_test_sets(df, target_column="Action", test_size=0.1, random_state=42, 
                                     shuffle=True, stratify=True, base_save_path="./datasets/"):
    
    
    if stratify:
        y_stratify = df[target_column].to_numpy()
    else:
        y_stratify = None
    
    train_df, test_df = train_test_split(df, 
                                         test_size=test_size, 
                                         random_state=random_state, 
                                         shuffle=shuffle, 
                                         stratify=y_stratify)
    
    timestamp = time.strftime("%Y%m%d_%H%M",time.localtime())
    train_save_path = os.path.join(base_save_path, f"train_{timestamp}.csv")
    test_save_path = os.path.join(base_save_path, f"test_{timestamp}.csv")
    
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    
    return train_df, test_df


def get_all_baseline_model_performance(df, target_column, estimators, metrics, n_cv_splits=5, random_state=7742, 
                                       shuffle=True, n_jobs=50, return_estimator=False, sort_metric="test_f1_weighted", 
                                       smaller_is_better=False, candidate_ohe_columns=['Source Port', 'Destination Port', 
                                                                                       'NAT Source Port', 'NAT Destination Port']):
    
    all_combinations = [list(c) for length in range(1, len(candidate_ohe_columns)+1) 
                        for c in combinations(candidate_ohe_columns, r=length)]
    
    all_combinations = [[]] + all_combinations
    
    result_dataframes = []
    
    for ohe_columns in all_combinations:
        
        # Drop the "port" columns which are not being one hot encoded
        drop_columns = [c for c in candidate_ohe_columns if c not in ohe_columns]
        model_df = df.drop(columns=drop_columns).copy(deep=True)

        features = [c for c in model_df.columns if c != target_column]

        # Scale the features that are not one hot encoded.
        scale_features = [c for c in features if c not in ohe_columns]

        # If we have things to ohe on this iteration, include ohe in the preprocess
        if ohe_columns != []:        

            preprocess = ColumnTransformer(transformers=[("ohe", OneHotEncoder(handle_unknown='infrequent_if_exist'), ohe_columns), 
                                                         ("scale", StandardScaler(), scale_features)], 
                                                          remainder="passthrough",  
                                                          n_jobs=n_jobs)
        else: # Else, no need for ohe in the preprocess

            preprocess = ColumnTransformer(transformers=[("scale", StandardScaler(), scale_features)], 
                                                          remainder="passthrough",  
                                                          n_jobs=n_jobs)
        
        models = [Pipeline(steps=[("preprocess", preprocess), ("model", model)]) for 
                    model in estimators]
            
        columns_encoded = ", ".join(ohe_columns)
            
        perf_df = get_baseline_model_performance(df=model_df, 
                                                 target_column=target_column, 
                                                 estimators=models, 
                                                 metrics=metrics, 
                                                 n_cv_splits=n_cv_splits, 
                                                 random_state=random_state, 
                                                 shuffle=shuffle, 
                                                 n_jobs=n_jobs, 
                                                 return_estimator=return_estimator, 
                                                 sort_metric=sort_metric, 
                                                 smaller_is_better=smaller_is_better)
            
        perf_df['columns_encoded'] = columns_encoded
        result_dataframes.append(perf_df)
            
    all_performance_dfs = pd.concat(objs=result_dataframes, 
                                    axis="index").sort_values(by=f"{sort_metric}_mean", 
                                                              ascending=smaller_is_better).reset_index(drop=True)
    
    
    first_columns = ["model", "columns_encoded"]
    column_order = first_columns + [c for c in all_performance_dfs.columns if c not in first_columns]
    
    return all_performance_dfs.loc[:, column_order]
            

def get_baseline_model_performance(df, target_column, estimators, metrics, n_cv_splits=10, random_state=42, 
                                   shuffle=True, n_jobs=30, return_estimator=False, sort_metric="test_f1_weighted", 
                                   smaller_is_better=False):
    
    baseline_results = get_baseline_model_results(df=df, 
                                                  target_column=target_column, 
                                                  estimators=estimators, 
                                                  metrics=metrics, 
                                                  n_cv_splits=n_cv_splits, 
                                                  random_state=random_state, 
                                                  shuffle=shuffle, 
                                                  n_jobs=n_jobs, 
                                                  return_estimator=return_estimator)
    
    performance_df = create_baseline_performance_df(baseline_results=baseline_results, 
                                                    metric=sort_metric, 
                                                    smaller_is_better=smaller_is_better)
    
    return performance_df

def get_baseline_model_results(df, target_column, estimators, metrics, n_cv_splits=10, random_state=42, 
                               shuffle=True, n_jobs=30, return_estimator=True):
    
    model_df = df.copy(deep=True)
    
    X = model_df.drop(columns=target_column)
    y = model_df[target_column].to_numpy()

    all_results = {}
    for estimator in estimators:

        if isinstance(estimator, Pipeline):
            name = str(estimator.named_steps['model']).replace("()", "")
        else:
            name = str(estimator).replace("()", "")

        cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=shuffle, random_state=random_state)
            
        all_results[name] = cross_validate(X=X, y=y, 
                                           estimator=estimator, 
                                           return_estimator=return_estimator,
                                           cv=cv, 
                                           n_jobs=n_jobs, 
                                           return_train_score=True, 
                                           error_score="raise", 
                                           scoring=metrics)
    return all_results


def create_baseline_performance_df(baseline_results, metric="test_f1", smaller_is_better=False, add_time=True):

    model_names = baseline_results.keys()
    dfs = []
    
    funcs = [('mean',np.mean), ('min',np.min), ('max',np.max), ('std', np.std)]
    
    for model_name in model_names:
        
        cols = [key for key in baseline_results[model_name].keys() 
                if key.startswith("train") or key.startswith("test")]

        if add_time:
            time_cols = [key for key in baseline_results[model_name].keys() if "time" in key]
            cols = cols + time_cols
        
        summary_df = pd.DataFrame({f"{metric}_{agg}":[func(baseline_results[model_name][metric])] 
                                   for metric in cols for agg, func in funcs})
        
        metric_cols = summary_df.columns.tolist()
        
        summary_df["model"] = model_name
        
        column_order = ["model"]+metric_cols
        
        summary_df = summary_df.loc[:,column_order]
        
        dfs.append(summary_df)
          
    result_df = pd.concat(objs=dfs).sort_values(by=f"{metric}_mean", 
                                                ascending=smaller_is_better).reset_index(drop=True)
    
    return result_df

############################## End Baseline Models ##############################

############################## Modeling ##############################

def run_gridsearch(X, y, estimator, param_grid, cv=None, n_jobs=-1, scoring=None, refit=None, verbose=3, 
                   error_score="raise", return_train_score=True, save=True, save_name=None, base_save_path="./models/", folds=5, 
                   shuffle=True, random_state=42):
    

    default_cv = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    cv = cv if cv is not None else default_cv

    scoring, refit, save_path = get_gs_args(estimator=estimator,
                                            scoring=scoring, 
                                            refit=refit, 
                                            save_name=save_name, 
                                            base_save_path=base_save_path)
    
    
    gs = GridSearchCV(estimator=estimator, 
                      param_grid=param_grid, 
                      scoring=scoring, 
                      refit=refit, 
                      n_jobs=n_jobs, 
                      cv=cv, 
                      verbose=verbose, 
                      error_score=error_score,
                      return_train_score=return_train_score)
    
    gs.fit(X,y)
    
    # Save gridsearch results with pickle
    if save:
        os.makedirs(base_save_path, exist_ok=True)
        with open(save_path, 'wb') as file:
            pickle.dump(gs, file)
    
    return gs

def get_gs_save_name(model_name):
    timestamp = time.strftime("%Y%m%d_%H%M",time.localtime())
    save_name = f"{model_name}_{timestamp}.pkl"
    return save_name

def load_gs_from_pickle(pickle_filepath):
    with open(pickle_filepath, 'rb') as file:
        results = pickle.load(file)
    return results

def get_gs_args(estimator, scoring, refit, save_name, base_save_path):
    
    # Set up defaults
    #
    default_scoring = ['f1', 
                       'balanced_accuracy', 
                       'roc_auc', 
                       'accuracy', 
                       'recall', 
                       'precision']

    timestamp = time.strftime("%Y%m%d_%H%M",time.localtime())
    default_save_name = f"{str(estimator).replace('()', '')}_{timestamp}_gs.pkl"
    
    # Setting args to either user supplied or default
    scoring = scoring if scoring is not None else default_scoring
    refit = refit if refit is not None else scoring[0]
    save_name = save_name if save_name is not None else default_save_name
    
    save_path_gs = os.path.join(base_save_path, save_name)
    
    return scoring, refit, save_path_gs

def error_analysis(model, X, y, dataset_type=""):
    
    dataset_type = dataset_type if dataset_type == "" else " " + dataset_type + " "
    
    pred_df = pd.DataFrame({"true":y, 
                            "predicted":model.predict(X)})
    
    pred_df["correct"] = (pred_df["predicted"] == pred_df["true"]).astype(int)
    
    true_counts = pred_df["true"].value_counts()
    true_pcts = pred_df["true"].value_counts(normalize=True)
    
    pred_counts = pred_df["predicted"].value_counts()
    pred_pcts = pred_df["predicted"].value_counts(normalize=True)
    
    correct_counts = pred_df.loc[pred_df["correct"]==1, "true"].value_counts()
    correct_pcts = pred_df.loc[pred_df["correct"]==1, "true"].value_counts(normalize=True)
    
    wrong_counts = pred_df.loc[pred_df["correct"]==0, "true"].value_counts()
    wrong_pcts = pred_df.loc[pred_df["correct"]==0, "true"].value_counts(normalize=True)
    
    ttop = f"============================ Breadown of True{dataset_type}Class Labels ============================"
    ptop = f"============================ Breadown of Predicted{dataset_type}Class Labels ============================"
    ctop = f"============================ Breadown of Correct{dataset_type}Predictions ============================"
    wtop =  f"============================ Breadown of Incorrect{dataset_type}Predictions ============================"
    
    # True
    print(ttop)
    print(true_counts, "\n\n", true_pcts, "\n", "="*len(ttop), "\n")
    
    # Predicted
    print(ptop)
    print(pred_counts, "\n\n", pred_pcts, "\n", "="*len(ptop), "\n")
    
    # Correct
    print(ctop)
    print(correct_counts, "\n\n", correct_pcts, "\n", "="*len(ctop), "\n")
    
    # Incorrect
    print(wtop)
    print(wrong_counts, "\n\n", wrong_pcts, "\n", "="*len(ctop), "\n")
    
    for target_class in np.unique(y):
        class_df = pred_df.loc[pred_df["true"]==target_class,:]
        class_pred_counts = class_df["predicted"].value_counts()
        class_pred_pcts = class_df["predicted"].value_counts(normalize=True)
        top = f"============================ Breadown of Predictions when true label is {target_class} ============================"
        print(top)
        print(class_pred_counts, "\n\n", class_pred_pcts, "\n", "="*len(top), "\n")
    
    return pred_df

def gs_to_clean_df(search_results, task="classification", sort_metric=None, sort_ascending=False):
    
    gs_df = pd.DataFrame(search_results)
    
    start_column_names = gs_df.columns
    
    # Remove the columns that give statistics on time or specific cv fold splits
    filtered_column_names = [name for name in start_column_names if "time" not in name]
    filtered_column_names = [name for name in filtered_column_names if "split" not in name]
    
    # Remove columns not in the filtered list above
    columns_to_remove = [name for name in start_column_names if name not in filtered_column_names]
    gs_df.drop(columns=columns_to_remove, inplace=True)
    
    # Columns we want to keep. Remainder of the function just fixes up these columns.
    column_names = gs_df.columns
    modified_column_names = [name.split("__")[-1] for name in column_names]
    modified_column_names = [name.split("param_")[-1] for name in modified_column_names]
    
    # For any negative metrics, take the absolute value and remove negative from the name.
    for col_name in modified_column_names:
        if "_neg" in col_name:
            gs_df.loc[:, col_name] = gs_df.loc[:, col_name].abs()
    modified_column_names = [name.replace("_neg", "") for name in modified_column_names]
    
    # Shorten some names for easier readability
    if task == "regression":
        shortened_names = [("_root_mean_squared_error", "_RMSE"), ("_mean_squared_error", "_MSE"), ("_mean_absolute_error","_MAE")]
        for long_name, short_name in shortened_names:
            modified_column_names = [name.replace(long_name, short_name) for name in modified_column_names]
    
    # Perform the final renaming
    renaming_dict = {old_name:new_name for old_name, new_name in zip(column_names, modified_column_names)}
    gs_df.rename(columns=renaming_dict, inplace=True)

    if sort_metric is None:

        if task == "regression":
            gs_df.sort_values(by="mean_test_RMSE", inplace=True)
        else:
            gs_df.sort_values(by="mean_test_accuracy", ascending=False, inplace=True)

    else:
        gs_df.sort_values(by=sort_metric, ascending=sort_ascending, inplace=True)

    return(gs_df)
############################## End Modeling ##############################



############################## Model Evaluation ##############################
def plot_confusion_matrix_from_estimator(df, target, model, format_digits_title=4, annot_fontsize=18, figsize=(7,7), 
                                         cmap="plasma", display_labels=None, colorbar=False, save_path=None,
                                         annot_shift=0.175, format_digits_annots=2, annot_horizontal_align="center", 
                                         title_fontsize=14, xlab_fontsize=16, ylab_fontsize=16, xlab_fontweight="bold", 
                                         ylab_fontweight="bold", tick_labelsize=11, title_x=0.575, title_y=0.915,
                                         title="Confusion Matrix: Training Set Predictions", title_weight="bold", 
                                         title_ha="center", title_va="center"):
    
    dataframe = df.copy(deep=True)
    
    X = dataframe.drop(columns=target)
    y = dataframe[target].to_numpy()
    
    preds = model.predict(X)
    
    dataframe['predicted'] = preds
    
    acc = accuracy_score(y_true=y, y_pred=preds)
    acc_str = f"{acc*100:.{format_digits_title}f}%"
    
    f1 = f1_score(y_true=y, y_pred=preds)
    f1_str = f"{f1:.{format_digits_title}f}"
    
    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : annot_fontsize}
    
    plt.rc('font', **font)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    
    matrix = ConfusionMatrixDisplay.from_estimator(estimator=model, 
                                                   X=X, 
                                                   y=y,                                                  
                                                   cmap=cmap, 
                                                   labels=[1,0], 
                                                   display_labels=display_labels, 
                                                   colorbar=colorbar,
                                                   ax=ax)
    
    total_positive_obs = np.sum([int(t.get_text()) for t in matrix.text_.ravel()[:2]])
    total_negative_obs = np.sum([int(t.get_text()) for t in matrix.text_.ravel()[2:]])

    positive_annots = [(t.get_position()[0], 
                        t.get_position()[1]+annot_shift, 
                        f"{(int(t.get_text())/total_positive_obs)*100:.{format_digits_annots}f}%", 
                        t.get_c()) for t in matrix.text_.ravel()[:2]]
    
    negative_annots = [(t.get_position()[0], 
                        t.get_position()[1]+annot_shift, 
                        f"{(int(t.get_text())/total_negative_obs)*100:.{format_digits_annots}f}%", 
                        t.get_c()) 
                       for t in matrix.text_.ravel()[2:]]
    
    new_annots = positive_annots+negative_annots
    
    for x, y, txt, c in new_annots:
        ax.annotate(txt, (x,y), color=c, horizontalalignment=annot_horizontal_align)
        
    ax.set_title(f"F1-Score={f1_str},  Accuracy={acc_str}", fontsize=title_fontsize, weight=title_weight)
    ax.set_xlabel("Predicted Class", fontsize=xlab_fontsize, weight=xlab_fontweight)
    ax.set_ylabel("True Class", fontsize=ylab_fontsize, weight=ylab_fontweight)
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    plt.suptitle(title, 
                 fontsize=title_fontsize, 
                 x=title_x,
                 y=title_y,
                 weight=title_weight, 
                 ha=title_ha, 
                 va=title_va)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    return ax


def add_newlines(label, every=20):
    new_label = ""
    break_due = False
    for index, c in enumerate(label):
        if index % every == 0:
            break_due = True
        if break_due and c == "_":
            new_label += f"\n{c}"
            break_due = False
        else:
            new_label += c
    return new_label

def plot_lgbm_feature_importance_from_gs(gs, importance_type="split", num_features=5, figsize=(18, 12), grid=False, bar_height=0.75, 
                                         title_fontsize=24, xlab_fontsize=20, ylab_fontsize=20, 
                                         feature_name_break_every=20, ytick_fontsize=14, save_path=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    ax = plot_importance(booster=gs.best_estimator_.named_steps['model'], 
                         max_num_features=num_features, 
                         ax=ax, 
                         importance_type=importance_type,
                         grid=grid,
                         height=bar_height)
    
    
    features = gs.best_estimator_[:-1].get_feature_names_out()
    column_numbers = [int(t.get_text().split("_")[-1]) for t in ax.get_yticklabels()[::-1]]
    top_feature_names = features[column_numbers]
    clean_labels = [add_newlines(label=feature, every=feature_name_break_every) 
                    for feature in top_feature_names]
    
    ax.set_title(f"Top {num_features} Most Important Features\nPer LightGBM '{importance_type}' importance type", fontsize=title_fontsize, weight="bold")
    ax.set_ylabel("Features", fontsize=xlab_fontsize)
    ax.set_xlabel("Importance", fontsize=ylab_fontsize)
    ax.set_yticklabels(labels=clean_labels, 
                       fontdict=dict(fontsize=ytick_fontsize))
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return ax

############################## End Model Evaluation ##############################