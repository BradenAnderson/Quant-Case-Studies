import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, make_scorer, fbeta_score
from lightgbm import LGBMClassifier, plot_importance, early_stopping
# from xgboost import XGBClassifier

############################## Data Preparation ##############################

def convert_strings_to_category(df):
    string_columns = df.select_dtypes(include=[object]).columns.tolist()
    df.loc[:,string_columns] = df.loc[:,string_columns].astype("category")
    return df

def create_train_val_test_sets(df, target_column="y", test_size=0.1, val_size=0.1, random_state=7742):
    
    train_df, val_and_test_df = train_test_split(df, 
                                                 test_size=test_size + val_size, 
                                                 random_state=random_state, 
                                                 stratify=df[target_column].to_numpy())
    
    val_df, test_df = train_test_split(val_and_test_df, 
                                       test_size=test_size / (test_size + val_size), 
                                       random_state=random_state, 
                                       stratify=val_and_test_df[target_column].to_numpy())
    
    print(f"Train Shape: {train_df.shape}")
    print(f"Val Shape: {val_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    
    return train_df, val_df, test_df

def subset_to_dai_selected_features(X_train, X_val, X_test):
    
    top_features = ["x48", "x23", "x27", 
                    "x20", "x28", "x46", 
                    "x49", "x37", "x42",
                    "x12", "x32", "x7", 
                    "x2", "x38", "x41", "x6", "x40"]
    
    X_train_dai = X_train.loc[:, top_features]
    X_val_dai = X_val.loc[:, top_features]
    X_test_dai = X_test.loc[:, top_features]
    
    print(f"DAI Train Shape: {X_train_dai.shape}")
    print(f"DAI Val Shape: {X_val_dai.shape}")
    print(f"DAI Test Shape: {X_test_dai.shape}")
    
    return X_train_dai, X_val_dai, X_test_dai

############################## End Data Preparation ##############################

############################## Modeling ##############################

def evaluate_baseline_model(X_train, y_train, X_val, y_val, model_pipe, model_name, scorers_dict):
    top = f"========== {model_name} =========="
    print(top)
    for score_type, scorer in scorers_dict.items():
        val_score = scorer(model_pipe, X_val, y_val)
        train_score = scorer(model_pipe, X_train, y_train)    
        print(f"Training {score_type}: {train_score}")
        print(f"Validation {score_type}: {val_score}\n")
    bottom = "="*len(top)
    print(bottom)
    return

def average_dollars_scorer_lgbm(y_true,  y_pred_probs):
    eval_name="avg_dollars_lost_per_prediction" 
    is_higher_better=False
    
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)
    
    eval_result = np.where(y_true==y_pred, 0, # Zero dollars lost for correct predictions
                           np.where(y_pred>y_true, 100, # 100 dollars lost when predict=1 and true=0
                                    np.where(y_pred<y_true, 20, 0))).sum() / y_pred.shape[0]
    

    return eval_name, eval_result, is_higher_better

def get_lgbm_best_iter_sklearn(estimator, X, y):
    
    if isinstance(estimator, Pipeline):
        return float(estimator.named_steps['model'].best_iteration_)
    else:
        return float(estimator.best_iteration_)
    

def average_dollars_scorer_sklearn(y, y_pred):
    
    eval_result = np.where(y==y_pred, 0, # Zero dollars lost for correct predictions
                           np.where(y_pred>y, 100, # 100 dollars lost when predict=1 and true=0
                                    np.where(y_pred<y, 20, 0))).sum() / y.shape[0]
    
    return eval_result

def fpt5_scorer_lgbm(y_true,  y_pred_probs):
    eval_name="f05_score" 
    is_higher_better=True
    
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)
    
    eval_result = fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5)
    
    return eval_name, eval_result, is_higher_better

def get_val_performance_after_all_train_refit(gs, X_train, y_train, X_val, y_val, model_name=""):
    
    scorers = {"accuracy":make_scorer(accuracy_score), 
               "f025_score":make_scorer(fbeta_score, beta=0.25),
               "f05_score":make_scorer(fbeta_score, beta=0.5), 
               "avg_dollars_lost_per_prediction":make_scorer(score_func=average_dollars_scorer_sklearn, 
                                                              greater_is_better=False)}
    
    
    
    if not isinstance(gs, Pipeline):
        is_lgbm =  isinstance(gs.best_estimator_.named_steps['model'], LGBMClassifier)
        if isinstance(gs.best_estimator_, Pipeline) and is_lgbm:
            best_iter = gs.best_estimator_.named_steps['model'].best_iteration_
            best_iter_string = f"Best Iteration: {best_iter}"
        elif is_lgbm:
            best_iter = gs.best_estimator_.best_iteration_
            best_iter_string = f"Best Iteration: {best_iter}"
        else:
            best_iter_string = ""
    else:
        best_iter_string = ""
    
    scores = {key:{"train":scorer(gs, X_train, y_train), 
                   "validation":scorer(gs, X_val, y_val)} 
              for key, scorer in scorers.items()}

    val_pred_df = pd.DataFrame({"true":y_val, 
                                "predicted":gs.predict(X_val)})
    
    train_pred_df = pd.DataFrame({"true":y_train, 
                                  "predicted":gs.predict(X_train)})
    
    top = "================="
    start = top + f" {model_name} " + top
    print(start)
    print(best_iter_string)
    for score, result in scores.items():
        train = result['train'] if result['train'] > 0 else result['train']*-1
        val = result['validation'] if result['validation'] > 0 else result['validation']*-1
        print(f"{score}: Train={train}, Validation={val}")
    bottom = "="*len(start)
    print(bottom)
    
    return train_pred_df, val_pred_df

def display_cross_validation_scores(gs, display_columns=None, model_name=""):
    
    default_display_columns = ["mean_test_f025_score", "mean_train_f025_score", 
                   "mean_test_f025_score", "mean_train_f025_score",
                   "mean_test_avg_dollars_lost_per_prediction", "mean_train_avg_dollars_lost_per_prediction",
                   "mean_test_accuracy", "mean_train_accuracy"]
    
    # "mean_test_best_iter_early_stopping"
    
    display_columns = display_columns if display_columns is not None else default_display_columns
    
    gs_df = pd.DataFrame(gs.cv_results_)
    
    gs_df = gs_df.loc[:, display_columns].T
    
    column = [model_name]
    
    if len(column) < len(gs_df.columns.tolist()):
        column = [f"{model_name} {index}" for index in range(1, len(gs_df.columns)+1)]
    
    gs_df.columns = column
    
    return gs_df

def run_gridsearch(X, y, estimator, param_grid, cv=None, n_jobs=-1, scoring=None, refit=None, verbose=3, 
                   error_score="raise", 
                   return_train_score=True, save=True, save_name=None, base_save_path="./models/", folds=5, 
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


def plot_confusion_matrix_from_predictions(pred_df, format_digits_title=4, annot_fontsize=18, figsize=(7,7), 
                                           cmap="plasma", display_labels=None, colorbar=False, save_path=None,
                                           annot_shift=0.175, format_digits_annots=2, annot_horizontal_align="center", 
                                           title_fontsize=14, xlab_fontsize=16, ylab_fontsize=16, xlab_fontweight="bold", 
                                           ylab_fontweight="bold", tick_labelsize=11, title_x=0.575, title_y=0.915,
                                           title="Confusion Matrix: Training Set Predictions", title_weight="bold", 
                                           title_ha="center", title_va="center"):
    
    preds = pred_df["predicted"].to_numpy()
    y = pred_df["true"].to_numpy()
    
    acc = accuracy_score(y_true=y, y_pred=preds)
    acc_str = f"{acc*100:.{format_digits_title}f}%"
    
    f1 = f1_score(y_true=y, y_pred=preds)
    f1_str = f"{f1:.{format_digits_title}f}"
    
    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : annot_fontsize}
    
    plt.rc('font', **font)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    
    matrix = ConfusionMatrixDisplay.from_predictions(y_true=y, 
                                                     y_pred=preds,                                               
                                                     cmap=cmap, 
                                                     values_format='d',
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
############################## End Model Evaluation ##############################