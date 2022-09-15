
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import os
import math
import scipy.stats as stats
from statsmodels.stats.proportion import proportion_confint
import time

#import tensorflow as tf

# Light GBM Models
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

# Sklearn preprocessing and data handling
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Regression models
from sklearn.svm import SVR  # dont use on large datasets
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Classification models
from sklearn.svm import SVC  # dont use on large datasets
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


def get_gs_args(estimator, scoring, refit, save_name, base_save_path):
    
    # Set up defaults
    #
    default_scoring = ["accuracy", "f1_weighted", "precision_weighted"]
    timestamp = time.strftime("%Y%m%d_%H%M",time.localtime())
    default_save_name = f"{str(estimator).replace('()', '')}_{timestamp}_gs.pkl"
    
    # Setting args to either user supplied or default
    scoring = scoring if scoring is not None else default_scoring
    refit = refit if refit is not None else scoring[0]
    save_name = save_name if save_name is not None else default_save_name
    
    save_path_gs = os.path.join(base_save_path, save_name)
    
    return scoring, refit, save_path_gs

def run_gridsearch(X, y, estimator, param_grid, cv=5, n_jobs=-1, scoring=None, refit=None, verbose=3, error_score="raise", 
                   return_train_score=True, save=True, save_name=None, base_save_path="./models/"):
    
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
    save_name = f"{model_name}_{timestamp}_gs.pkl"
    return save_name

def load_gs_from_pickle(pickle_filepath):
    with open(pickle_filepath, 'rb') as file:
        results = pickle.load(file)
    return results

############################################### PLOTTING FUNCTIONS ###############################################


def plot_corr_matrix(dataframe, top_n=None, target=None, figsize=(10, 10), cmap="mako", annotate=True, linewidths=1):
    
    if top_n and target:
        print(f"Creating matrix for top {top_n} features most correlated with {target}")
        correlations = dataframe.corr()
        top_corr_feature_names = list(correlations.loc[:, [target]].sort_values(by=target, ascending=False).index[:top_n+1])
        dataframe = dataframe.loc[:, top_corr_feature_names]
        
    
    # Get a dataframe of correlations
    correlations = dataframe.corr()
    
    # Create a mask to remove the redundant entries
    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True
    
    # Create the axis for plotting
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    
    sns.heatmap(correlations, cmap = 'mako', annot = annotate, square = True, vmin = -1, vmax = 1, mask = mask, linewidths=linewidths, ax=axs);


# Plot the distribution of categorical features, and the percentage of observations in each level.
def plot_categorical_features(dataframe, features_list, num_cols=3, title_fontsize=16,
                              xlab_fontsize=12, ylab_fontsize=12, tick_fontsize=12, tick_rotation=45,
                              annot_fontsize=10, round_digits=4, add_annotations=True,):

    num_rows = math.ceil(len(features_list) / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 6 * num_rows), squeeze=False)
    
    # For each plot we are going to add
    for index, plot_name in enumerate(features_list):
        
        # Grab the grid location for this plot
        col = index % num_cols
        row = index // num_cols
        
        # Add the plot
        sns.countplot(data=dataframe, x=plot_name, ax=axs[row][col])
        
        # Annotate plot axes
        axs[row][col].set_title(f"{plot_name}", fontsize=title_fontsize, weight='bold')
        axs[row][col].set_xlabel(plot_name, fontsize=xlab_fontsize, weight='bold')
        axs[row][col].set_ylabel(f"Count of {plot_name} by category", fontsize=ylab_fontsize, weight='bold')
        axs[row][col].tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)

        if add_annotations:

            # Total people number of observations
            total = len(dataframe.index)

            # Annotate the percentages on top of the bars
            for p in axs[row][col].patches: 
            
                # Percentage is the ratio of the bar height over the total people
                percentage = f"{round((100 * (p.get_height() / total)), round_digits)}%"

            
                # Annotate on the left edge of the bar
                x = p.get_x()
            
                # Annotate just above the top of the bar
                y = p.get_y() + p.get_height() + 0.04
            
                #Perform annotation
                axs[row][col].annotate(percentage, (x,y), fontsize=annot_fontsize, fontweight='bold')

    plt.tight_layout()


def plot_categorical_means(df, grouping_variables_list, response_variable, conf_interval="sd", num_cols=2, 
                           xlab_fontsize=14, ylab_fontsize=14, tick_fontsize=12, tick_rotation=45, title_fontsize=16):

    sns.set_style("darkgrid")
    
    num_rows = math.ceil(len(grouping_variables_list) / num_cols)
    
    if num_rows == 1:
        num_cols = len(grouping_variables_list)
        
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10 * num_cols, 6 * num_rows), squeeze=False)
    
    for index, grouping_variable in enumerate(grouping_variables_list):
        
        # Grab the grid location for this plot
        col = index % num_cols
        row = index // num_cols
        
        # Order groups smallest to largest according to their associated mean response
        x_label_order = list(df.groupby(by=grouping_variable)[response_variable].mean().sort_values().index)
        
        # Create the means plot
        sns.pointplot(x=grouping_variable, y=response_variable, data=df, ci=conf_interval, order=x_label_order, ax=axs[row][col])
        
        # Add plot labels
        plot_title = f"Mean of {response_variable} at each level of {grouping_variable}"
        axs[row][col].set_title(f"{plot_title}", fontsize=title_fontsize, weight='bold')
        axs[row][col].set_xlabel(grouping_variable, fontsize=xlab_fontsize, weight='bold')
        axs[row][col].set_ylabel(f"Mean of {response_variable}", fontsize=ylab_fontsize, weight='bold')
        axs[row][col].tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
        
    plt.tight_layout()
    
    return plt.show()


def calculate_proportion_plot_data(df, grouping_variable, response_variable, response_success_level, conf_level):
    
    # Count the number of occurences of each possible response, for each level of the grouping variable
    counts =  df.groupby(by=grouping_variable)[response_variable].value_counts()
    
    # Total Number of observations in each level of the group variable
    group_var_num_obs = counts.groupby(by=grouping_variable).sum()
    
    # List of levels in the grouping variable
    group_var_levels = list(group_var_num_obs.index)
    
    # Filter the value counts so we only have counts for "successes" (where the grouping variable was associated
    # with the response_variable having the level specified in response_success_level).
    success_counts = counts[counts.index[counts.index.get_level_values(response_variable) == response_success_level]]
    
    # Confidence interval for the proportion of responses that have "response_success_level" for each level of the grouping variable
    proportion_cis = [(group_level, *proportion_confint(count=success_counts[(group_level, response_success_level)], 
                                                        nobs=group_var_num_obs[group_level], alpha=conf_level)) for group_level in group_var_levels]
    
    
    # Add the center of the confidence interval (best estimate) to the list above, which contains the group name, upper and lower bounds
    proportions_with_center = sorted([(name, lower, ((upper - lower)/2)+lower, upper) for name, lower, upper in proportion_cis], key=lambda sublist: sublist[2])
    
    # Separate out the X coordinates (names), y coordinates (best_estimates) and error_bar values
    names = [name for name, lower, center, upper in proportions_with_center]
    best_estimates = [center for name, lower, center, upper in proportions_with_center]
    error_bars=[best_estimate - lower for name, lower, best_estimate, upper in proportions_with_center]
    
    return {"x_coords":names, "y_coords":best_estimates, "y_error_amounts":error_bars}


def plot_categorical_proportions(df, grouping_variable_list, response_variable, response_success_level, conf_level=0.05, num_cols=2, 
                                 xlab_fontsize=14, ylab_fontsize=14, tick_fontsize=12, tick_rotation=45, title_fontsize=16):
    
    sns.set_style("darkgrid")
    
    num_rows = math.ceil(len(grouping_variable_list) / num_cols)
    
    if num_rows == 1:
        num_cols = len(grouping_variable_list)
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10 * num_cols, 6 * num_rows), squeeze=False)
    
    for index, grouping_variable in enumerate(grouping_variable_list):
        
        # Grab the grid location for this plot
        col = index % num_cols
        row = index // num_cols
        
        plot_data = calculate_proportion_plot_data(df=df, 
                                                   grouping_variable=grouping_variable, 
                                                   response_variable=response_variable, 
                                                   response_success_level=response_success_level,
                                                   conf_level=conf_level)
        
        axs[row][col].errorbar(x=plot_data["x_coords"], y=plot_data["y_coords"], yerr=plot_data["y_error_amounts"])
        
        
        plot_title = f"Proportion of {response_variable} having value of {response_success_level}\nat each level of {grouping_variable}"
        axs[row][col].set_title(f"{plot_title}", fontsize=title_fontsize, weight='bold')
        axs[row][col].set_xlabel(grouping_variable, fontsize=xlab_fontsize, weight='bold')
        axs[row][col].set_ylabel(f"Proportion of {response_variable}={response_success_level}", fontsize=ylab_fontsize, weight='bold')
        axs[row][col].tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
    
    plt.tight_layout()
    
    return plt.show()


def plot_continuous_features(dataframe, features_list, num_cols=3, round_digits=4, title_fontsize=16,
                             xlab_fontsize=12, ylab_fontsize=12, tick_fontsize=12, tick_rotation=45, 
                             hist_stat="frequency", kde=False, binwidth=None):

    sns.set_style("darkgrid")
    
    num_rows = math.ceil(len(features_list) / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 6 * num_rows), squeeze=False)
    
    for index, plot_name in enumerate(features_list):
    
        col = index % num_cols
        row = index // num_cols
    
        sns.histplot(data=dataframe, x=plot_name, ax=axs[row][col], stat=hist_stat, kde=kde, binwidth=binwidth)
        
        # Calcualte metrics for plot title
        distribution_array = dataframe.loc[:, plot_name].to_numpy()
        skew = stats.skew(distribution_array)
        kurtosis = stats.kurtosis(distribution_array)

        # Create plot title
        plot_title = (f"{plot_name}\n"
                      f"Skew: {round(skew, round_digits)} Kurtosis: {round(kurtosis, round_digits)}")
        
        # Add plot title and axis labels
        axs[row][col].set_title(plot_title, fontsize=title_fontsize, weight='bold')
        axs[row][col].set_xlabel(plot_name, fontsize=xlab_fontsize, weight='bold')
        axs[row][col].set_ylabel(f"{hist_stat} of {plot_name}", fontsize=ylab_fontsize, weight='bold')
        axs[row][col].tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)

    
    plt.tight_layout()


#def plot_missing_values(df, figsize=(16,8), bar_color="dodgerblue", bar_sort="ascending"):
#    
#    columns_with_missings = list(df.isna().sum()[df.isna().sum() > 0].index)
#    
#    missing_df = df.loc[:, columns_with_missings]
#    
#    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
#    
#    msno.bar(missing_df, color=bar_color, sort=bar_sort, ax=axs)
#    
#    return plt.show()

############################################### END PLOTTING FUNCTIONS ###############################################


############################################### SKLEARN FUNCTIONS ###############################################


def create_column_transformer(numeric_features, numeric_pipe=None, ordinal_cat_feats=None, ordinal_pipe=None, nominal_cat_feats=None, 
                              nominal_pipe=None, numeric_impute_strategy="median", categorical_impute_strategy="most_frequent",
                              remainder_columns="drop", ordinal_categories="auto", sparse_threshold=0.3):
    
    # Instantiate the pipelines for:
    # 1. Numeric Features, 2. Ordinal Categorical Features, 3. Nominal Categorical Featuers
    # by either using the pipeline passed into the function, or creating a simple default pipeline for that type of data.
    if numeric_pipe is None: 
        numeric_pipeline = Pipeline([('num_imputer', SimpleImputer(strategy=numeric_impute_strategy)),
                                     ('std_scaler', StandardScaler())])
    else:
        numeric_pipeline = numeric_pipe
    
    if ordinal_pipe is None:
        ordinal_pipeline = Pipeline([('ord_imputer', SimpleImputer(strategy=categorical_impute_strategy)),
                                     ('ord_encoder',  OrdinalEncoder(categories=ordinal_categories))])
    else:
        ordinal_pipeline = ordinal_pipe


    if nominal_pipe is None:
        nominal_pipeline = Pipeline([('nom_imputer', SimpleImputer(strategy=categorical_impute_strategy)), 
                                     ('nom_encoder', OneHotEncoder())])
    else:
        nominal_pipeline = nominal_pipe


    # If there are no features for a particular data type, set the features to an empty list
    # that way the associated sub-pipeline in the ColumnTransformer doesn't transform anything
    if numeric_features is None:
        numeric_features = []
    
    if ordinal_cat_feats is None:
        ordinal_cat_feats = []
    
    if nominal_cat_feats is None:
        nominal_cat_feats = []
    
    # Create the combined pipeline that transforms each data type.
    full_pipeline = ColumnTransformer(transformers=[("ord_cat_feats", ordinal_pipeline, ordinal_cat_feats),
                                                    ("numeric_feats", numeric_pipeline, numeric_features), 
                                                    ("nom_cat_feats", nominal_pipeline, nominal_cat_feats)],
                                     remainder=remainder_columns,
                                     sparse_threshold=sparse_threshold)
    
    return full_pipeline


# This function evaluates multiple Sklearn models (can be regression or classification) 
# All models are evaluated via k-fold cross validation 
# All models are fit using their default hyperparameters
#
# The purpose of this function is to quickly be able to short list a few model types for further
# investigation and hyperparameter tuning.
#
# Valid task types --> regression, classification, multiclass_classification
# 
def cv_base_models(X, y, estimator_list=None, score_metrics=None, cv_folds=5, n_jobs=-1, include_train_score=True,
                   provide_status_every=2, task_type="regression"):
    
    if estimator_list is None:
        estimator_list = get_default_cv_estimators(task_type=task_type)
        
    if score_metrics is None:
        score_metrics = get_default_metrics_list(task_type=task_type)
    
    for model_number, (model, model_name) in enumerate(estimator_list, start=1):
        
        cv_result = cross_validate(estimator=model,
                                   X=X,
                                   y=y, 
                                   scoring=score_metrics, 
                                   cv=cv_folds, 
                                   n_jobs=n_jobs,
                                   return_train_score=include_train_score)
        
        # Get the keys in the cv_result dictionary that correspond to metrics we care about
        # and set up our dictionary of results
        if model_number == 1:
            metric_keys = [name for name in cv_result.keys() if name.startswith("test") or name.startswith("train")]
            
            # Set up dictionary of results
            results = {name:[] for name in metric_keys}
            results["model_name"] = []
            results["cv_folds"] = [cv_folds] * len(estimator_list)
        
        # Update dictionary of results
        for metric_name in metric_keys:
            scores = cv_result[metric_name]
            
            if "neg" in metric_name:
                scores = scores * -1
            
            results[metric_name].append(np.mean(scores))
        
        results["model_name"].append(model_name)
            
        if model_number % provide_status_every == 0:
            print("========================================================")
            print(f"Finished model {model_number}, {model_name}")
            print("========================================================\n\n")
            
    result_df = create_base_model_cv_df(results=results, task_type=task_type)
        
    return result_df

def create_base_model_cv_df(results, task_type):
    
    result_df = pd.DataFrame(results)
    
    if task_type == "regression":
        
        result_df.columns = [name.replace("_neg", "") for name in result_df.columns]
        
        if "test_root_mean_squared_error" in result_df.columns:
            result_df.sort_values(by="test_mean_squared_error", inplace=True)
    
    else:
        if "test_accuracy" in result_df.columns:
            result_df.sort_values(by="test_accuracy", ascending=False, inplace=True)
    
    return result_df

def get_default_cv_estimators(task_type):
    
    if task_type == "regression":
        estimator_list = [(SVR(), "sklearn.svm.SVR"),   # Remove this if dataset is very large. Expensive to train.
                          (LinearSVR(), "sklearn.svm.LinearSVR"),
                          (RandomForestRegressor(), "sklearn.ensemble.RandomForestRegressor"),
                          (ExtraTreesRegressor(), "sklearn.ensemble.ExtraTreesRegressor"),
                          (LGBMRegressor(boosting_type="gbdt"), "LGBM_GBDT"),
                          (LGBMRegressor(boosting_type="dart"), "LGBM_DART"),
                          (KNeighborsRegressor(), "KNN"),
                          (LinearRegression(), "LinearRegression"),
                          (Lasso(), "Lasso")]
        
    elif task_type == "classification" or "multiclass_classification":
        estimator_list = [(SVC(), "sklearn.svm.SVC"),   # Remove this if dataset is very large. Expensive to train.
                          (LinearSVC(), "sklearn.svm.LinearSVC"),
                          (RandomForestClassifier(), "sklearn.ensemble.RandomForestClassifier"),
                          (ExtraTreesClassifier(), "sklearn.ensemble.ExtraTreesClassifier"),
                          (LGBMClassifier(boosting_type="gbdt"), "LGBM_GBDT"),
                          (LGBMClassifier(boosting_type="dart"), "LGBM_DART"),
                          (AdaBoostClassifier(), "sklearn.ensemble.AdaBoostClassifier"),
                          (KNeighborsClassifier(), "KNN"),
                          (LogisticRegression(), "LogisticRegression"),
                          (RidgeClassifier(), "sklearn.linear_model.RidgeClassifier")]
        
    return estimator_list


### GridSearch Related Functions
def get_default_metrics_list(task_type):
        
    if task_type == "regression":
        return ["neg_root_mean_squared_error", "neg_mean_squared_error", "neg_mean_absolute_error"]

    elif task_type == "classification":
        return ["accuracy", "f1", "precision", "recall"]
    
    elif task_type == "multiclass_classification":
        return ["accuracy", "f1_macro", "f1_micro", "precision_macro", "precision_micro", "recall_macro", "recall_micro"]

# Default Hyperparameters for a GridSearch on LightGBMRegressor
def get_lgbm_reg_default_param_grid():
    
    grid = {'num_leaves':[21, 31, 41],          # Main param for controlling complexity. Default is 31.
            'min_child_samples':[10, 20, 30],    # Important for overfitting. Large value may cause underfitting. Best depends on min_leaves. Default is 20.
            'max_depth':[-1, 10],
            'n_estimators':[100, 300, 900],
            'learning_rate':[0.03, 0.1, 0.3],
            'colsample_bytree':[0.9, 1]}        # Subsample ratio of columns. Default is 1.
    
    return grid


def get_svm_reg_default_param_grid():
    
    grid = {'kernel':["poly", "rbf"],
            'epsilon':[0.05, 0.1, 0.5], # Default is 0.1
            'C':[0.3, 1, 3]}

    return grid

def get_svm_clf_default_param_grid():

    grid = {'kernel':["sigmoid","poly", "rbf"],
            'C':[0.3, 1, 3],
            'gamma':['scale', 'auto']}
    
    return grid


def get_linear_svm_reg_default_param_grid():

    grid = {'C':[0.3, 1, 3, 10],
            'loss':['epsilon_insensitive', 'squared_epsilon_insensitive']}

    return grid

def get_knn_reg_default_param_grid():

    grid = {'n_neighbors':[2, 5, 10, 15, 20, 25, 30, 50],
            'weights': ['uniform', 'distance']}
    
    return grid

def get_rf_reg_default_param_grid():

    grid = {'n_estimators':[100, 300],
            'min_samples_leaf': [1, 5, 10],
            'min_samples_split':[2, 5]}
    
    return grid

def get_ada_boost_clf_default_param_grid():

    grid = {'n_estimators': [50, 150, 500],
            'learning_rate': [1.0, 0.5, 0.2]}
    return grid

def get_default_param_grid(model_type_string):
    
    if model_type_string == "<class 'lightgbm.sklearn.LGBMRegressor'>":
        return get_lgbm_reg_default_param_grid()
    elif model_type_string == "<class 'lightgbm.sklearn.LGBMClassifier'>":
        return get_lgbm_reg_default_param_grid()    # Same defaults as LGBMRegressor for now
    elif model_type_string == "<class 'sklearn.svm._classes.SVR'>":
        return get_svm_reg_default_param_grid()
    elif model_type_string == "<class 'sklearn.svm._classes.LinearSVR'>":
        return get_linear_svm_reg_default_param_grid()
    elif model_type_string == "<class 'sklearn.neighbors._regression.KNeighborsRegressor'>":
        return get_knn_reg_default_param_grid()
    elif model_type_string == "<class 'sklearn.ensemble._forest.RandomForestRegressor'>":
        return get_rf_reg_default_param_grid()
    elif model_type_string == "<class 'sklearn.ensemble._forest.ExtraTreesRegressor'>": 
        return get_rf_reg_default_param_grid() # Extra trees regressor same as random forest regressor for now.
    elif model_type_string == "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>":
        return get_rf_reg_default_param_grid() # Extra trees classifier same as random forest regressor for now.
    elif model_type_string == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
        return get_rf_reg_default_param_grid() # RF classifier same as random forest regressor for now.
    elif model_type_string == "<class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>":
        return get_ada_boost_clf_default_param_grid()
    elif model_type_string == "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>":
        return get_knn_reg_default_param_grid() ## KNN clf same as KNN reg for now.
    elif model_type_string == "<class 'sklearn.svm._classes.SVC'>":
        return get_svm_clf_default_param_grid() 

    return -1



def perform_gridsearch(X, y, model, parameter_grid=None, cv_folds=5, njobs=-1, metrics=None, task="regression", 
                       refit_metric=None, verbose=0, return_train_score=False, save_name=None, base_save_path="./models/"):
    
    # Get a default param_grid if one has been specified in get_default_param_grid
    if parameter_grid is None:
        param_grid = -1
        param_grid = get_default_param_grid(model_type_string=str(type(model)))
        if param_grid == -1:
            print("Error. No param_grid passed, and no default grid for this model type.")
    else:
        param_grid = parameter_grid
    
    # Get a default list of metrics, depending on if task="regression" or task="classification"
    if metrics is None:
        metrics = get_default_metrics_list(task_type=task)
        
    # metric for determining the best model found by gridsearch, so gridsearch can
    # refit using this model at the very end. Refitting trains on all available data.
    # This metric is the first in the metrics list, unless otherwise specified in the function arguments.
    if refit_metric is None:
        refit_metric = metrics[0]
    
    
    # Instantiate the GridSearch
    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      scoring=metrics,
                      refit=refit_metric,
                      n_jobs=njobs, 
                      cv=cv_folds,
                      verbose=verbose, 
                      return_train_score=return_train_score)
    
    # Perform the GridSearch
    gs.fit(X, y)
    
    # If desired, save the gridsearch results
    # requires setting the save_name parameter
    if save_name:
        os.makedirs(base_save_path, exist_ok=True)
        save_path = os.path.join(base_save_path, save_name)
        
        with open(save_path, 'wb') as file:
            pickle.dump(gs, file)
        
    return gs


def gs_to_clean_df(search_results, task="regression", sort_metric=None, sort_ascending=True):
    
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


def get_test_set_metrics(model, X_test, y_test, task_type="regression"):
    
    y_pred = model.predict(X_test)
    
    test_metrics = {}
    
    if task_type == "regression":
        test_metrics["RMSE"] = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
        test_metrics["MSE"] = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=True)
        test_metrics["MAE"] = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        
    elif task_type == "classification":
        test_metrics["accuracy"] = accuracy_score(y_true=y_test, y_pred=y_pred)
        test_metrics["f1"] = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
        test_metrics["precision"] = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
        test_metrics["recall"] = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
        test_metrics["auc_roc"] = roc_auc_score(y_true=y_test, y_pred=y_pred)
        
    elif task_type == "multiclass_classification":
        test_metrics["accuracy"] = accuracy_score(y_true=y_test, y_pred=y_pred)
        test_metrics["f1_macro"] = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
        test_metrics["f1_micro"] = f1_score(y_true=y_test, y_pred=y_pred, average="micro")
        test_metrics["precision_macro"] = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
        test_metrics["precision_micro"] = precision_score(y_true=y_test, y_pred=y_pred, average="micro")
        test_metrics["recall_macro"] = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
        test_metrics["recall_micro"] = recall_score(y_true=y_test, y_pred=y_pred, average="micro")
    
    return test_metrics

def display_correct_num_output_columns(df, numeric_features, nominal_cat_feats, ordinal_cat_feats):
    
    num_columns = len(numeric_features) + len(ordinal_cat_feats)
    
    for nominal_feature_name in nominal_cat_feats:
        num_columns += len(df.loc[:, nominal_feature_name].unique())
    
    print(f"Output should have {num_columns} after preprocessing")
    
    return num_columns

#### Deep Learning Functions
def get_learning_rate_scheduler(initial_lr=1e-7, lr_multiplication_factor=10, lr_multiply_every=20):

    # Create a learning rate scheduler to adjust the learning rate during training
    #
    # Start with a very small learning rate (initial_lr) e.g. 1x10^-6 or 1x10^-7
    #
    # Increase learning rate gradually, by increasing by a factor
    # of lr_multiplication_factor every lr_multiply_every epochs.
    #
    # Pass this lr_schedule as a callback to the fit method to adjust lr during training

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: initial_lr * lr_multiplication_factor**(epoch/lr_multiply_every)
    )

    return lr_schedule