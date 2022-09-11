import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from lightgbm import LGBMRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFECV

import pickle
import time
import os


########################################################### DATA CLEANING FUNCTIONS ###########################################################

def get_repeat_columns(train_csv_df, unique_m_csv_df):
    repeat_columns = [col for col in train_csv_df.columns if col in unique_m_csv_df.columns]
    print(f"Columns found in both dataframes: {repeat_columns}, duplicates will be removed prior to dataframe concatenation.\n")
    return repeat_columns
    
def drop_single_value_columns(dataframe):
    nuniques = dataframe.nunique()
    single_value_columns = nuniques[nuniques == 1].index.tolist()
    print(f"There are {len(single_value_columns)} columns that contain the exact same value for every row: {single_value_columns}")
    before_drop_shape = dataframe.shape
    dataframe = dataframe.drop(columns=single_value_columns)
    after_drop_shape = dataframe.shape
    print(f"Shape before dropping single value columns: {before_drop_shape}, shape after dropping: {after_drop_shape}\n")
    return dataframe

def merge_and_clean(train_csv_df, unique_m_csv_df):
    
    print(f"========================== Preprocessing Steps ==========================\n")
    duplicate_columns = get_repeat_columns(train_csv_df=train_csv_df, unique_m_csv_df=unique_m_csv_df)
    
    dataframe = pd.concat(objs=[train_csv_df, 
                                unique_m_csv_df.drop(columns=duplicate_columns)], 
                          axis="columns")
    
    dataframe = drop_single_value_columns(dataframe=dataframe)
    
    print(f"=========================================================================")
    return dataframe

########################################################### END DATA CLEANING FUNCTIONS ###########################################################



########################################################### BASELINE MODELING FUNCTIONS ###########################################################

def get_model_name(estimator, estimator_named_step="model"):
    if isinstance(estimator, Pipeline):
        estimator_name = str(estimator.named_steps[estimator_named_step])
    else:
        estimator_name = str(estimator)
    return estimator_name.replace("()","")

def get_baseline_model_results(X, y, estimators, metrics, n_cv_splits=5, random_state=42, shuffle=True, n_jobs=-1, fspec=5, return_estimator=True):
    
    cv = KFold(n_splits=n_cv_splits, shuffle=shuffle, random_state=random_state)
    name_map = {"neg_mean_squared_error":"MSE", "neg_root_mean_squared_error":"RMSE", "neg_mean_absolute_error":"MAE", "r2":"R-Squared"}
    
    print(f"============================== Baseline Models: {n_cv_splits}-Fold CV Metrics (All Default Hyperparameters) ============================================")
    print("Null Model (No parameters, always predicts the mean): ")
    print(get_dumb_baseline(y=y),"\n")
    
    all_results = {}
    
    for estimator in estimators:
        estimator_name = get_model_name(estimator=estimator)
        print(f">>>>>>>>>>>>>>> {estimator_name} <<<<<<<<<<<<<<<")
            
        cv_results = cross_validate(X=X, 
                                    y=y, 
                                    estimator=estimator, 
                                    return_estimator=return_estimator,
                                    cv=cv, 
                                    n_jobs=n_jobs, 
                                    return_train_score=True, 
                                    error_score="raise", 
                                    scoring=metrics)
        
        # Number of non-zero coefficients
        num_coefs = [len(est.named_steps["model"].coef_[est.named_steps["model"].coef_ != 0]) for est in cv_results["estimator"]]
        min_coefs = np.min(num_coefs)
        max_coefs = np.max(num_coefs)
        avg_coefs = np.mean(num_coefs)
        print(f"Number of Non-Zero Model Parameter Estimates:")
        print(f"Avg: {avg_coefs}, Min: {min_coefs}, Max: {max_coefs}\n")

        test_metrics = {name_map.get(key.replace("test_",""), key):((np.mean(value), np.min(value), np.max(value), np.std(value)) if "neg" not in key 
                        else (np.mean(value*-1), np.min(value*-1), np.max(value*-1), np.std(value*-1))) for key, value in cv_results.items() if "test" in key}

        train_metrics = {name_map.get(key.replace("train_",""), key):((np.mean(value), np.min(value), np.max(value), np.std(value)) if "neg" not in key else 
                        (np.mean(value*-1), np.min(value*-1), np.max(value*-1), np.std(value*-1))) for key, value in cv_results.items() if "train" in key}
        
        all_results[estimator_name]={"cv":cv_results, "train":train_metrics, "test":test_metrics, "estimator":estimator}
        for metric_name, (avg_val, min_val, max_val, std)in test_metrics.items():
            if train_metrics.get(metric_name, None) is not None:
                train = (f"Avg Train {metric_name}:{train_metrics[metric_name][0]:.{fspec}f}, min:{train_metrics[metric_name][1]:.{fspec}f} "
                         f"max:{train_metrics[metric_name][2]:.{fspec}f} std:{train_metrics[metric_name][3]:.{fspec}f}")
                print(train)
            print(f"Avg Validation {metric_name}: {avg_val:.{fspec}f}, min:{min_val:.{fspec}f}, max:{max_val:.{fspec}f} std:{std:.{fspec}f}\n")

    print("====================================================================================================================================")
    return all_results

def get_dumb_baseline(y, fspec=5):
    
    dumb_baseline_prediction = np.mean(y)
    baseline_preds = [dumb_baseline_prediction for _ in range(len(y))]
    
    mse = mean_squared_error(y_true=y, y_pred=baseline_preds, squared=True)
    rmse = mean_squared_error(y_true=y, y_pred=baseline_preds, squared=False)
    mae = mean_absolute_error(y_true=y, y_pred=baseline_preds)
    r2 = r2_score(y_true=y, y_pred=baseline_preds)
    
    result = (f"MSE: {mse:.{fspec}f}\nRMSE: {rmse:{fspec}f}\nMAE: {mae:{fspec}f}\n"
              f"R-Squared: {r2:{fspec}f} (Always zero for constant model that predicts the mean)")
    
    return result

def get_baseline_estimators(random_state=42):

    lr_pipe = Pipeline(steps=[("scaler", StandardScaler()), 
                              ("model", LinearRegression())])

    lasso_pipe = Pipeline(steps=[("scaler", StandardScaler()), 
                                 ("model", Lasso(random_state=random_state))])

    ridge_pipe = Pipeline(steps=[("scaler", StandardScaler()), 
                                 ("model", Ridge(random_state=random_state))])

    enet_pipe = Pipeline(steps=[("scaler", StandardScaler()), 
                                ("model", ElasticNet(random_state=random_state))])

    return [lr_pipe, lasso_pipe, ridge_pipe, enet_pipe]

########################################################### END BASELINE MODELING FUNCTIONS ###########################################################


########################################################### LASSO MODEL IMPORTANCE FUNCTIONS ###########################################################
def get_lasso_parameter_df(X, y, alpha=1.0, max_iter=5_000, tol=1e-4, random_state=42, positive=False, print_summary=True, model_type="lasso", l1_ratio=None):
    
    if model_type == "lasso":
        model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state, positive=positive)
    elif model_type == "elasticnet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=max_iter, tol=tol)
    
    model.fit(X,y)

    lasso_coef_df = pd.DataFrame({"Parameter":model.coef_, "Feature":model.feature_names_in_}, index=model.feature_names_in_)
    lasso_coef_df["Parameter_Rank"] = ((lasso_coef_df["Parameter"].abs().to_numpy()*-1).argsort().argsort() + 1)
    lasso_coef_df = lasso_coef_df.reindex(lasso_coef_df["Parameter"].abs().sort_values(ascending=False).index)
    lasso_coef_df.reset_index(drop=True, inplace=True)
    if print_summary:
        desc = lasso_coef_df["Parameter"].describe()
        eliminated_features = lasso_coef_df.loc[lasso_coef_df["Parameter"]==0,"Feature"].tolist()
        top = f"============================== {model_type.title()} Parameter Estimate Summary (Alpha={alpha})"
        if l1_ratio:
            top += f" (l1_ratio = {l1_ratio})"
        top+=" ======================================="
        print(top)
        print(f"Number of Parameters Estimated: {desc['count']}")
        print(f"Max Parameter Value: {desc['max']}")
        print(f"75th Percentile Parameter Value: {desc['75%']}")
        print(f"Mean Parameter Value: {desc['mean']}")
        print(f"Median Parameter Value: {desc['50%']}")
        print(f"Min Parameter Value: {desc['min']}\n")
        print(f"Number of non-zero parameter estimates: {X.shape[1] - len(eliminated_features)}")
        print(f"Number of Parameters Eliminated by {model_type.title()} (coef shrank to zero): {len(eliminated_features)}\n")
        print(f"Features Eliminated By Lasso:\n{', '.join(eliminated_features)}")
        bottom = "="*len(top)
        print(bottom)
    lasso_selected_df = lasso_coef_df.loc[lasso_coef_df["Parameter"]!=0,:]
    return lasso_coef_df, lasso_selected_df

def get_vif_df(X):
    return pd.DataFrame(data={"VIF":[variance_inflation_factor(exog=X, exog_idx=index) for index in range(X.shape[1])]}, 
                        index=X.columns.to_numpy())

def get_statsmodels_fit_info(X, y, features_used_str=None, subset_columns=None):
    
    X_fit = X.copy(deep=True)
    if subset_columns is not None:
        X_fit = X_fit.loc[:,subset_columns]
    
    ols = OLS(endog=y, 
              exog=add_constant(X_fit.copy(deep=True)), 
              hasconst=True).fit()
    
    vif_df = get_vif_df(X=X_fit)

    sm_results = {"ols_coef":ols.params.tolist(), 
                  "pval":ols.pvalues.tolist(), 
                  "pval_rank":["CONST TO REMOVE"] + (ols.pvalues.to_numpy()[1:].argsort().argsort() + 1).tolist(),
                  "lwr_ci":ols.conf_int()[0].tolist(), 
                  "upr_ci":ols.conf_int()[1].tolist(), 
                  "Feature":["const"]+X_fit.columns.tolist()}
    
    if subset_columns is not None:
        not_used_columns = [colname for colname in X.columns if colname not in subset_columns]
        for key, value in sm_results.items():
            if key == "Feature":
                sm_results[key] = value + not_used_columns
            else:
                sm_results[key] = value + ["NOT_USED" for _ in range(len(not_used_columns))]

    sm_df = pd.DataFrame(sm_results, index=sm_results["Feature"])
    sm_df.loc[:,vif_df.columns] = vif_df
    sm_df.reset_index(drop=True) 
    sm_df = sm_df.loc[sm_df["Feature"] != "const",:]
    
    if features_used_str is not None:
        sm_df.rename(columns={name:f"{name}_{features_used_str}" for name in sm_df.columns if name != "Feature"}, inplace=True)
        
    return sm_df

def get_lightgbm_importances_df(X, y):
    lgbm_reg = LGBMRegressor(random_state=42, importance_type="split")
    lgbm_reg.fit(X, y)

    lgbm_split_df = pd.DataFrame({"lgbm_importance_split":lgbm_reg.feature_importances_, 
                            "lgbm_split_rank":((np.array(lgbm_reg.feature_importances_)*-1).argsort().argsort()+1)}, index=lgbm_reg.feature_name_)

    lgbm_reg = LGBMRegressor(random_state=42, importance_type="gain")
    lgbm_reg.fit(X, y)

    lgbm_gain_df = pd.DataFrame({"lgbm_importance_gain":lgbm_reg.feature_importances_, 
                            "lgbm_gain_rank":((np.array(lgbm_reg.feature_importances_)*-1).argsort().argsort()+1)}, index=lgbm_reg.feature_name_)

    lgbm_df = lgbm_split_df.copy(deep=True)
    lgbm_df.loc[:,lgbm_gain_df.columns] = lgbm_gain_df
    return lgbm_df

def get_mutual_info_df(X, y, discrete_feature=False, random_state=42):
    mi = mutual_info_regression(X=X, y=y, discrete_features=False, random_state=42)
    mi_ranks = (mi*-1).argsort().argsort() + 1
    mi_df = pd.DataFrame({"mutual_info":mi, "mutual_info_rank":mi_ranks,"Feature":X.columns})
    mi_df.set_index(keys="Feature", drop=True, inplace=True)
    return mi_df

def get_rfecv_mse_rmse_df(X, y, min_features_to_select=1, random_state=42, n_jobs=-1, shuffle=True, n_splits=5):
    
    mse_df = get_rfecv_df(X, y, 
                          min_features_to_select=min_features_to_select, 
                          random_state=random_state, 
                          n_jobs=n_jobs, 
                          shuffle=shuffle, 
                          n_splits=n_splits, 
                          scoring="neg_mean_squared_error")
    
    
    rmse_df = get_rfecv_df(X, y, 
                          min_features_to_select=min_features_to_select, 
                          random_state=random_state, 
                          n_jobs=n_jobs, 
                          shuffle=shuffle, 
                          n_splits=n_splits, 
                           scoring="neg_root_mean_squared_error")
    
    rfecv_df = mse_df.copy(deep=True)
    rfecv_df.loc[:,rmse_df.columns] = rmse_df
    return rfecv_df

def get_rfecv_mse_rmse_df(X, y, min_features_to_select=1, random_state=42, n_jobs=-1, shuffle=True, n_splits=5):
    
    mse_df = get_rfecv_df(X, y, 
                          min_features_to_select=min_features_to_select, 
                          random_state=random_state, 
                          n_jobs=n_jobs, 
                          shuffle=shuffle, 
                          n_splits=n_splits, 
                          scoring="neg_mean_squared_error")
    mse_df.rename(columns={"RFECV_rank":"RFECV_mse_rank"}, inplace=True)
    
    
    rmse_df = get_rfecv_df(X, y, 
                          min_features_to_select=min_features_to_select, 
                          random_state=random_state, 
                          n_jobs=n_jobs, 
                          shuffle=shuffle, 
                          n_splits=n_splits, 
                           scoring="neg_root_mean_squared_error")
    rmse_df.rename(columns={"RFECV_rank":"RFECV_rmse_rank"}, inplace=True)
    
    rfecv_df = mse_df.copy(deep=True)
    rfecv_df.loc[:,rmse_df.columns] = rmse_df
    return rfecv_df

def get_rfecv_df(X, y, min_features_to_select=1, random_state=42, n_jobs=-1, shuffle=True, n_splits=5, 
                 scoring="neg_root_mean_squared_error"):
    
    recursive_select = RFECV(estimator=LinearRegression(), 
                             min_features_to_select=min_features_to_select, 
                             cv=KFold(n_splits=n_splits, 
                                      shuffle=shuffle, 
                                      random_state=random_state), 
                             scoring=scoring, 
                             n_jobs=n_jobs)
    
    recursive_select.fit(X=X, y=y)
    
    rfecv_df = pd.DataFrame({"Feature":recursive_select.feature_names_in_, "RFECV_rank":recursive_select.ranking_})
    rfecv_df.set_index(keys="Feature", drop=True, inplace=True)
    return rfecv_df

def get_lasso_plot_df(X, y, X_unscaled=None, alpha=1.0, max_iter=20_000, tol=1e-4, random_state=42, positive=False, print_summary=True, 
                      return_selected_only=False, mi_discrete_feature=False, min_features_to_select=1, n_jobs=-1, shuffle=True, run_rfecv=False, 
                      model_type="lasso", l1_ratio=None):
    
    X_lgbm = X_unscaled if X_unscaled is not None else X

    sm_all_df = get_statsmodels_fit_info(X=X, y=y, features_used_str="all")
    
    lasso_all_df, lasso_selected_df = get_lasso_parameter_df(X=X, y=y, 
                                                             alpha=alpha,
                                                             max_iter=max_iter, 
                                                             tol=tol, 
                                                             random_state=random_state, 
                                                             positive=positive, 
                                                             print_summary=print_summary, 
                                                             model_type=model_type, 
                                                             l1_ratio=l1_ratio)
    
    sm_selected_df = get_statsmodels_fit_info(X=X, y=y, features_used_str="selected", subset_columns=lasso_selected_df.loc[:,"Feature"].tolist())
    
    lgbm_imp_df = get_lightgbm_importances_df(X=X_lgbm, y=y)
    mutual_info_df = get_mutual_info_df(X=X, y=y, discrete_feature=mi_discrete_feature)

    lasso_plot_df = lasso_all_df.copy(deep=True).set_index(keys="Feature")
    lasso_plot_df.loc[:,sm_selected_df.set_index(keys="Feature").columns] = sm_selected_df.set_index(keys="Feature")
    lasso_plot_df.loc[:,sm_all_df.set_index(keys="Feature").columns] = sm_all_df.set_index(keys="Feature")
    lasso_plot_df.loc[:,lgbm_imp_df.columns] = lgbm_imp_df
    lasso_plot_df.loc[:, mutual_info_df.columns] = mutual_info_df

    if run_rfecv:
        rfecv_df = get_rfecv_mse_rmse_df(X=X, y=y, 
                                        min_features_to_select=min_features_to_select, 
                                        random_state=random_state, 
                                        shuffle=shuffle,
                                        n_jobs=n_jobs)
        lasso_plot_df.loc[:,rfecv_df.columns] = rfecv_df

    lasso_plot_df.reset_index(inplace=True)
    
    lasso_plot_selected_df = lasso_plot_df.loc[lasso_plot_df["Parameter"]!=0,:]
    dtype_convert = {colname:float for colname in lasso_plot_selected_df.columns if type(lasso_plot_selected_df[colname].to_numpy()[0]) == float}
    lasso_plot_selected_df = lasso_plot_selected_df.astype(dtype_convert)
    
    if return_selected_only:
        return lasso_plot_selected_df

    return lasso_plot_df, lasso_plot_selected_df, sm_all_df, sm_selected_df, lasso_all_df, lasso_selected_df


def get_lasso_large_plot_df(X, y, alphas, X_unscaled=None, max_iter=10_000, tol=1e-4, random_state=42, positive=False, print_summary=True, run_rfecv=False):
    
    alpha_levels = []
    for index, alpha in enumerate(alphas):
        p_df = get_lasso_plot_df(X=X, y=y, X_unscaled=X_unscaled,
                                 max_iter=max_iter, 
                                 alpha=alpha,
                                 tol=tol, 
                                 random_state=random_state, 
                                 positive=positive, 
                                 print_summary=print_summary, 
                                 run_rfecv=run_rfecv,
                                 return_selected_only=True)
        
        alpha_level = f"alpha={alpha}"
        alpha_levels.append(alpha_level)

        p_df["regularization_strength"] = [alpha_level for _ in range(p_df.shape[0])]
        if index == 0:
            all_p_df = p_df.copy(deep=True)
        else:
            all_p_df = pd.concat(objs=[all_p_df, p_df])
    
    all_p_df["regularization_strength"] = pd.Categorical(all_p_df["regularization_strength"].tolist(), categories=alpha_levels)
    all_p_df["Parameter_abs"] = all_p_df["Parameter"].abs().to_numpy() 
    all_p_df.sort_values(by=["regularization_strength", "Parameter_abs", "Feature"], inplace=True)
    return all_p_df

########################################################### END LASSO MODEL IMPORTANCE FUNCTIONS ###########################################################

########################################################### ELASTIC NET PLOTTING FUNCTIONS ###########################################################

def find_closest_value(find_value, all_values):
    return all_values[np.abs(all_values - find_value).argmin()]

def get_is_best(value, best_train, best_test):
    if best_train != best_test:
        if value == best_train:
            return "best_train"
        elif value == best_test:
            return "best_test"
    elif best_train == best_test and value == best_test:
        return "best_train_and_test"

    return "not_best"

def create_elastic_net_plot_df(gs_dataframe, shade_column_values, metric_column_y, shade_column="alpha", 
                               include_best=True, best_is_biggest=False):
    
    shade_col_vals = gs_dataframe[shade_column].to_numpy()
    plot_shade_values = [find_closest_value(find_value=value, all_values=shade_col_vals) for value in shade_column_values]
    
    if "train" in metric_column_y:
        train_metric = metric_column_y
        test_metric = metric_column_y.replace("train", "test")
    else:
        train_metric = metric_column_y.replace("test", "train")
        test_metric = metric_column_y
    
    if include_best:
        biggest_train = gs_dataframe.loc[gs_dataframe[train_metric]==gs_dataframe[train_metric].max(),shade_column].to_numpy()[0]
        smallest_train = gs_dataframe.loc[gs_dataframe[train_metric]==gs_dataframe[train_metric].min(),shade_column].to_numpy()[0]
        best_train = biggest_train if best_is_biggest else smallest_train
        
        biggest_test = gs_dataframe.loc[gs_dataframe[test_metric]==gs_dataframe[test_metric].max(),shade_column].to_numpy()[0]
        smallest_test = gs_dataframe.loc[gs_dataframe[test_metric]==gs_dataframe[test_metric].min(),shade_column].to_numpy()[0]
        best_test = biggest_test if best_is_biggest else smallest_test
        
        if best_test == best_train:
            plot_shade_values = [best_test] + plot_shade_values
        else:
            plot_shade_values = [best_test, best_train] + plot_shade_values
        
        p_df = gs_dataframe.loc[gs_dataframe[shade_column].isin(plot_shade_values),:].copy(deep=True)
    
        p_df["is_best"] = [get_is_best(value, best_train, best_test) for value in p_df[shade_column].tolist()]

        return p_df

def get_hover_setting(X, column_name, float_format=':.4f', exclude_cols=None):
    if exclude_cols is not None:
        if column_name in exclude_cols:
            return False
    
    if type(X[column_name].to_numpy()[0]) == float:
        return float_format
    else:
        return True

def create_metrics_scatter_plotly(plot_df, metric_column_y, hyperparam_column_x="l1_ratio", shade_column="alpha", title=None, height=600, 
                                  width=1500, float_format=':.4f', exclude_from_hover_cols=None, symbol=None, xmax=None, xmin=None, trendline=None):
    
    if xmax:
        plot_df = plot_df.loc[plot_df[hyperparam_column_x] <= xmax,:]
    if xmin:
        plot_df = plot_df.loc[plot_df[hyperparam_column_x] >= xmin,:]

    metric_cols = [col for col in plot_df.columns if "test" in col or "train" in col]
    hover_cols = [metric_column_y, hyperparam_column_x, shade_column] + metric_cols
    hover_data = {col:get_hover_setting(X=plot_df, 
                                        column_name=col, 
                                        float_format=float_format, 
                                        exclude_cols=exclude_from_hover_cols) for col in hover_cols}
    
    shade_str = f",shaded by {shade_column}" if shade_column is not None else ""
    t = (f"5 fold Cross-Validation <b>{metric_column_y}</b> vs {hyperparam_column_x} {shade_str}<br>" 
     f"L1_Ratio=0 is a pure L2 penalty (Ridge), L1_Ratio=1 is a pure L1 penalty (Lasso)")
    
    fig = px.scatter(data_frame=plot_df, 
                     y=metric_column_y, 
                     x=hyperparam_column_x, 
                     color=shade_column,
                     hover_data=hover_data, 
                     trendline=trendline,
                     title=t, 
                     symbol=symbol,
                     height=height, 
                     width=width)
    return fig
########################################################### END ELASTIC NET PLOTTING FUNCTIONS ###########################################################

########################################################### GRIDSEARCH FUNCTIONS ###########################################################

def get_gs_args(estimator, scoring, refit, save_name, base_save_path):
    
    # Set up defaults
    #
    default_scoring = ["neg_root_mean_squared_error", "neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
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

def gs_to_clean_df(search_results, task="classification", sort_metric=None, sort_ascending=True):
    
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

########################################################### END GRIDSEARCH FUNCTIONS ###########################################################

def get_gs_alphas(df):
    
    alpha_df = df.loc[df["fit_intercept"]==True,:].copy(deep=True)
    
    min_f = alpha_df["mean_test_RMSE"]==alpha_df["mean_test_RMSE"].min()
    max_f = alpha_df["mean_test_RMSE"]==alpha_df["mean_test_RMSE"].max()
    median_f = alpha_df["mean_test_RMSE"]==alpha_df["mean_test_RMSE"].median()
    
    best_alpha = alpha_df.loc[min_f,"alpha"].to_numpy()[0]
    # worst_alpha = alpha_df.loc[max_f,"alpha"].to_numpy()[0]   # ways model will all zero param estimates
    middle_alpha = alpha_df.loc[median_f,"alpha"].to_numpy()[0]

    #bad_index = int(df.shape[0] * 0.75)
    #bad_alpha = df.sort_values(by="mean_test_RMSE").reset_index().iloc[bad_index]["alpha"]
    
    # return best_alpha, bad_alpha, middle_alpha
    return {'best':best_alpha, 'median':middle_alpha}


def get_plotly_train_test_compare_df(df, metric="RMSE", train_metric=None, test_metric=None):
    
    metric = metric if metric is not None else train_metric.replace("mean_train_", "")
    train_metric = train_metric if train_metric is not None else f"mean_train_{metric}"
    test_metric = test_metric if test_metric is not None else f"mean_test_{metric}"
    
    test_plot_df = df.copy(deep=True)
    test_plot_df["metric_name"] = [f"Test {metric}" for _ in range(test_plot_df.shape[0])]
    test_plot_df["metric_value"] = test_plot_df[test_metric].to_numpy()
    
    test_min = test_plot_df["metric_value"].min()
    test_max = test_plot_df["metric_value"].max()
    test_plot_df["test_min"] = ["test_min" if val==test_min else "" for val in test_plot_df["metric_value"].to_numpy()]
    test_plot_df["test_max"] = ["test_max" if val==test_max else "" for val in test_plot_df["metric_value"].to_numpy()]
    
    train_plot_df = df.copy(deep=True)
    train_plot_df["metric_name"] = [f"Train {metric}" for _ in range(train_plot_df.shape[0])]
    train_plot_df["metric_value"] = train_plot_df[train_metric].to_numpy()
    train_plot_df["test_min"] = ["no" for _ in train_plot_df["metric_value"].to_numpy()]
    train_plot_df["test_max"] = ["no" for _ in train_plot_df["metric_value"].to_numpy()]
    
    
    return pd.concat(objs=[test_plot_df,train_plot_df])


def get_plotly_train_test_compare_title(df, metric, hyperparam, annotate_min):
    test_metric = f"mean_test_{metric}"
    best_metric = df[test_metric].min() if annotate_min else df[test_metric].max()
    best_hyperparam = df.loc[df[test_metric]==best_metric,hyperparam].to_numpy()[0]
    
    t = (f"{metric} vs {hyperparam}, " 
         f"Best Test {metric} was {best_metric} at {hyperparam}={best_hyperparam}")
    return t

def plotly_train_test_compare(df, annotate_min=True, metric_y="RMSE", hyperparameter_x="alpha", train_metric=None, test_metric=None, 
                              height=600, width=1300, title=None, xmax=None, xmin=None, title_new_line=None, hover_columns="all", trendline=None):
    
    p_df = get_plotly_train_test_compare_df(df=df, metric=metric_y, train_metric=train_metric, test_metric=test_metric)
    labs = {"metric_value":metric_y}
    symbol = "test_min" if annotate_min else "test_max"
    
    if hover_columns == "all":
        hover_data = {col:True for col in df.columns}
    else:
        hover_data = None

    t = get_plotly_train_test_compare_title(df=p_df, metric=metric_y, hyperparam=hyperparameter_x, annotate_min=annotate_min)
    if title_new_line:
        t = t + f"<br>{title_new_line}"
    
    if xmax:
        p_df = p_df.loc[p_df[hyperparameter_x] <= xmax,:]
    if xmin:
        p_df = p_df.loc[p_df[hyperparameter_x] >= xmin,:]

    fig = px.scatter(data_frame=p_df, 
                     y="metric_value", 
                     x=hyperparameter_x, 
                     color="metric_name", 
                     hover_data=hover_data,
                     labels=labs, trendline=trendline,
                     symbol=symbol, 
                     category_orders={symbol:["no", symbol]},
                     symbol_sequence=["circle", "star"],
                     title=title if title is not None else t, 
                     height=height, 
                     width=width)
    
    return fig

def create_ridge_coef_plot_df(X, ridge_coef_df, lasso_coef_df):

    lcoef_df = lasso_coef_df.copy(deep=True)
    lcoef_df["lasso_eliminated"] = ["Lasso_Eliminated" if param_est == 0.0 else "" for param_est in lcoef_df["Parameter"].to_numpy()]
    lcoef_df.rename(columns={"Parameter":"Lasso_Parameter"}, inplace=True)
    lcoef_df.set_index(keys="Feature", drop=True, inplace=True)
    
    ridge_coef_df.set_index(keys="Feature", drop=True, inplace=True)
    ridge_coef_df.loc[:,lcoef_df.columns] = lcoef_df
    
    vif_df=get_vif_df(X)
    ridge_coef_df.loc[:,vif_df.columns] = vif_df
    ridge_coef_df.reset_index(drop=False, inplace=True)
    
    ridge_coef_df = ridge_coef_df.reindex(ridge_coef_df["Parameter"].abs().sort_values(ascending=True).index)
    return ridge_coef_df


    ### Ensemble
def get_sorted_coef_df(coefs, features, model_name):
    
    c_df = pd.DataFrame({"Feature":features,
                      "Parameter":coefs, 
                      "model":[model_name for _ in range(len(features))]})
    
    c_df = c_df.reindex(c_df["Parameter"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    
    c_df = c_df.reset_index().rename(columns={"index":"rank_in_model"})
    
    c_df["rank_in_model"] = c_df["rank_in_model"] + 1
    
    return c_df

def get_ensemble_coef_df(ensemble_pipe, ranking_model_name="lasso", num_top_features=10):
    
    ensemble_features = ensemble_pipe.feature_names_in_
    
    ensemble_coefs = {estimator_name:estimator.coef_ for estimator_name, estimator 
                      in ensemble_pipe.named_steps["ensemble"].named_estimators_.items()}
    
    coef_dfs = {estimator_name:get_sorted_coef_df(coefs=c, features=ensemble_features, model_name=estimator_name) 
                for estimator_name, c in ensemble_coefs.items()}

    ranking_df = coef_dfs[ranking_model_name].copy(deep=True)
    
    ranking_df = ranking_df.loc[ranking_df.index < num_top_features,:]
    top_feature_names = ranking_df["Feature"].tolist()
    del coef_dfs[ranking_model_name]
    
    subset_dataframes = [ranking_df] + [c_df.loc[c_df["Feature"].isin(top_feature_names),:] for c_df in coef_dfs.values()] 
    plot_df = pd.concat(objs=subset_dataframes, axis="index")
    
    plot_df["Feature_Order"] = pd.Categorical(plot_df["Feature"].to_numpy(), top_feature_names)
    plot_df.sort_values(by="Feature_Order", inplace=True, ascending=False)
    return plot_df.drop(columns="Feature_Order")

def get_ensemble_plot_info(ensemble_pipe,  num_top_features=10, ranking_model_name="lasso"):
    
    p_df = get_ensemble_coef_df(ensemble_pipe=ensemble_pipe,  
                                num_top_features=num_top_features, 
                                ranking_model_name=ranking_model_name)
    
    title = (f"Coefficients for ensemble of lasso, and ridge regression<br>" 
             f"Top {num_top_features} parameters from the {ranking_model_name} portion are displayed, along with the corresponding coefficients from the ridge model")
    
    return {"df":p_df, "title":title}

def get_important_features(df, metrics):
    important = []
    for metric in metrics:
        ind = df.sort_values(by=metric).index[:5]
        imp = df.loc[df.index.isin(ind),"Feature"].tolist()
        important.extend(imp)
    return important

def get_ensemble_model_df(model_df, rank_df, lasso_gs, metrics):
    
    important_features = get_important_features(df=rank_df, metrics=metrics)
    lasso_selected_features = get_lasso_selected_from_gs(lasso_gs=lasso_gs)
    poly_features = important_features + lasso_selected_features
    poly_df = get_polynomial_features_df(df=model_df.loc[:,poly_features])
    ensemble_df = pd.concat(objs=[model_df, poly_df], axis="columns")
    
    return ensemble_df

def get_important_features(df, metrics):
    important = []
    for metric in metrics:
        ind = df.sort_values(by=metric).index[:5]
        imp = df.loc[df.index.isin(ind),"Feature"].tolist()
        important.extend(imp)
    return important

def get_lasso_selected_from_gs(lasso_gs):
    features = lasso_gs.best_estimator_[:-1].get_feature_names_out()
    coefs = lasso_gs.best_estimator_.named_steps["model"].coef_
    selected_features = features[coefs != 0].tolist()
    return selected_features

def get_ensemble_model_df(model_df, rank_df, lasso_gs, metrics):
    
    important_features = get_important_features(df=rank_df, metrics=metrics)
    lasso_selected_features = get_lasso_selected_from_gs(lasso_gs=lasso_gs)
    
    poly_features = important_features + lasso_selected_features
    
    poly_df = get_polynomial_features_df(df=model_df.loc[:,poly_features])
    
    ensemble_df = pd.concat(objs=[model_df, poly_df], axis="columns")
    
    return ensemble_df, poly_features

def get_important_features(df, metrics):
    important = []
    for metric in metrics:
        ind = df.sort_values(by=metric).index[:5]
        imp = df.loc[df.index.isin(ind),"Feature"].tolist()
        important.extend(imp)
    return important

def get_lasso_selected_from_gs(lasso_gs):
    features = lasso_gs.best_estimator_[:-1].get_feature_names_out()
    coefs = lasso_gs.best_estimator_.named_steps["model"].coef_
    selected_features = features[coefs != 0].tolist()
    return selected_features

def get_polynomial_features_df(df, degree=(2,3), include_bias=False):
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(df)
    poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
    return poly_df

def plot_target_correlations(model_df, target="critical_temp", corr_method="pearson", figsize=(5,16), 
                             num_correlations=15, savepath=None, tick_fontsize=15, title_fontsize=16):
    
    corr = model_df.corr(method=corr_method)
    
    target_corr = corr.loc[:,[target]]
    
    target_corr = target_corr.reindex(target_corr[target].abs().sort_values(ascending=False).index)
    
    target_corr = target_corr[:num_correlations+1]
    
    target_corr = target_corr.loc[target_corr.index != target,:]
    
    
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(target_corr, 
                     cmap='crest', 
                     annot=True, 
                     vmin=-1, 
                     vmax=1, 
                     linewidth=0.1, 
                     linecolor='w', annot_kws={"fontsize":20})
    
    ax.set_title(f"Top {num_correlations} Features with largest absolute value of \nof {corr_method} correlation with {target}", 
                 fontsize=title_fontsize)
    
    ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath)
    
    return ax

def get_residuals(observed, predicted):
    return [obs - pred for obs, pred in zip(observed, predicted)]

def get_residplot_df(estimator, X, y):
    
    preds = estimator.predict(X)
    
    resids = get_residuals(observed=y, predicted=preds)
    
    resid_df = pd.DataFrame({"observed":y, "predicted":preds, "residuals":resids})
    
    return resid_df

def plot_residual_vs_fitted_values(plot_df, residual_column="residuals", fitted_columns="predicted", 
                                   figsize=(14, 5), title=None, model_name=None, title_fontsize=16, 
                                   tick_fontsize=10, tick_rotation=0, legend_frameon=False, 
                                   xlab_fontsize=12, ylab_fontsize=12, alpha=0.6, savepath=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.scatterplot(x=fitted_columns, 
                    y=residual_column, 
                    data=plot_df, 
                    alpha=alpha, 
                    ax=ax)
    
        # Annotate plot axes
    if title is None:
        if model_name is None:
            title = "Residuals vs Fitted Values"
        else:
            title = f"Residuals vs Fitted Values for {model_name}"
    
    ax.axhline(y=0, xmin=plot_df[fitted_columns].min(), 
               xmax=plot_df[fitted_columns].max(), c='black', linestyle ="--")
    ax.set_title(f"{title}", fontsize=title_fontsize, weight='bold')
    ax.set_xlabel("Fitted Values", fontsize=xlab_fontsize, weight='bold')
    ax.set_ylabel("Residuals", fontsize=ylab_fontsize, weight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
    #ax.legend(loc=legend_location, frameon=legend_frameon)
    
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath)
    return ax

def plot_observed_vs_predicted_values(plot_df, predicted_column="predicted", observed_column="observed", 
                                   figsize=(14, 5), title=None, model_name=None, title_fontsize=16, 
                                   tick_fontsize=10, tick_rotation=0, legend_frameon=False, 
                                   xlab_fontsize=12, ylab_fontsize=12, alpha=0.4, savepath=None, 
                                      ci=None, fit_reg=True, scatter=True):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.regplot(x=observed_column, 
                    y=predicted_column, 
                    data=plot_df, 
                    scatter_kws={"alpha":alpha}, 
                  ci=ci, 
                  scatter=scatter,
                    ax=ax, 
                  fit_reg=fit_reg)
    
        # Annotate plot axes
    if title is None:
        if model_name is None:
            title = "Observed vs Predicted Values"
        else:
            title = f"Observed vs Predicted Values for {model_name}"
    

    
    ax.set_title(f"{title}", fontsize=title_fontsize, weight='bold')
    ax.set_xlabel("Observed Values", fontsize=xlab_fontsize, weight='bold')
    ax.set_ylabel("Predicted Values", fontsize=ylab_fontsize, weight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
    #ax.legend(loc=legend_location, frameon=legend_frameon)
    
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath)
    return ax