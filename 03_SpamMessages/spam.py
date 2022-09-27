import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pickle
import time
import re
import os
import email

import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def get_label_from_dirname(dirpath, positive_indicator="spam"):
    if positive_indicator in dirpath:
        return 1
    else:
        return 0 
    
# https://docs.python.org/2.4/lib/standard-encodings.html
def import_messages(root_dir="./SpamAssassinMessages", encoding="cp437", positive_indicator="spam"):
    
    messages = {"message":[], "label":[], "foldername":[],
                "filename":[],"filepath":[]}
    
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for name in filenames:
            fullpath = os.path.join(dirpath, name)
            messages['label'].append(get_label_from_dirname(dirpath=dirpath, positive_indicator=positive_indicator))
            messages['foldername'].append(os.path.basename(dirpath))
            messages['filepath'].append(fullpath)
            messages['filename'].append(name)
            with open(fullpath,'r', encoding=encoding) as f:
                try:
                    msg = email.message_from_file(f)
                    messages['message'].append(msg)
                except UnicodeDecodeError as e:
                    print(f"Error occured with encoding type: {encoding}\n{e}")
                    return
                 
    return messages

def build_message_string(message):
    
    msg_text = ""
    for msg_part in message.walk():
        if "text" in msg_part.get_content_type():
            msg_text = msg_text + " " + msg_part.get_payload()
                
    return msg_text

def import_emails(root_dir="./SpamAssassinMessages", encoding="cp437", positive_indicator="spam"):
    
    messages = import_messages(root_dir=root_dir, 
                               encoding=encoding, 
                               positive_indicator=positive_indicator)
    
    messages['text'] = [build_message_string(message=msg) for msg in messages['message']]
    messages['is_multipart'] = [int(msg.is_multipart()) for msg in messages['message']]
    messages['content_type'] = [msg.get_content_type() for msg in messages['message']]
    messages['content_main_type'] = [msg.get_content_maintype() for msg in messages['message']]
    messages['content_sub_type'] = [msg.get_content_subtype() for msg in messages['message']]
    messages['charsets'] = [msg.get_content_subtype() for msg in messages['message']]
    messages['params'] = [msg.get_charsets() for msg in messages['message']]
    
    msg_df = pd.DataFrame(messages)
    first_cols = ['text', 'label', 'is_multipart', 'content_type']
    col_order =  first_cols + [c for c in msg_df.columns if c not in first_cols]
    msg_df = msg_df.loc[:, col_order]

    return msg_df

############################## Initial Preprocessing ##############################

def text_preprocessing(df, text_column="text", punctuation=None, custom_stop=None, nltk_stop=True, 
                       spacy_default_stop=True, all_stop=True, lemma=True):
    
    pp_df = df.copy(deep=True)
    
    # Remove punctuation
    remove_strategies = get_default_remove_strategies()
    punct_df = pp_df[[text_column]].apply(lambda row: normalize_text(row=row, remove_strategies=remove_strategies), axis="columns", result_type="expand")
    pp_df = pd.concat(objs=[pp_df, punct_df], axis="columns")

    # Remove contractions
    contraction_map = get_contraction_map()
    cont_df = pp_df.loc[:, list(remove_strategies.keys())].apply(lambda row: replace_contractions(row=row, 
                                                                                            cmap=contraction_map), 
                                                            axis="columns", 
                                                            result_type="expand")
    pp_df = pd.concat(objs=[pp_df, cont_df], axis="columns")

    # Lemmatization being performed separate for all removal strategies 
    nlp = spacy.load("en_core_web_lg")
    lem_cols = [c for c in pp_df.columns if c.startswith("text_clean")]
    lemma_df = pp_df.loc[:,lem_cols].apply(lambda row: lemmatize_email(row=row, nlp=nlp), 
                                           axis="columns", 
                                           result_type='expand')
    pp_lem_df = pd.concat(objs=[pp_df, lemma_df], axis="columns")

    # All stopword strategies being performed one at a time on each lemmatized column
    stopword_strategies = get_default_stopword_strategies()
    for txt_col in [col for col in pp_lem_df.columns if "_lem" in col]:
        stop_df = pp_lem_df.loc[:,[txt_col]].apply(lambda row: remove_stopwords(row=row, 
                                                                                nlp=nlp,
                                                                                ss=stopword_strategies), 
                                                    axis="columns", 
                                                    result_type="expand")

        pp_lem_df = pd.concat(objs=[pp_lem_df, stop_df], axis="columns")

    first_cols = [c for c in pp_lem_df.columns if c.startswith("text")]
    col_order = first_cols + [c for c in pp_lem_df.columns if c not in first_cols]

    return pp_lem_df.loc[:,col_order]

def normalize_text(row, remove_strategies, text_column="text"):
     return {key:" ".join([re.sub(remove_pattern, "", word.strip()) for word in row[text_column].lower().split()]) 
     for key, remove_pattern in remove_strategies.items()}

def lemmatize_email(row, nlp):
    return {f"{name}_lem":" ".join([tok.lemma_ for tok in nlp(row[name]) if not tok.is_space]) 
            for name in row.index.to_numpy()}

def remove_stopwords(row, ss, nlp):
    row_name = row.index.to_numpy()[0]
    return {f"{row_name}_{ss_type}":" ".join([tok.text for tok in nlp(row[row_name]) if tok.text not in stop]) 
                for ss_type, stop in ss.items()}
    
def replace_contractions(row, cmap):
    return {f"{name}c":" ".join([cmap.get(w, w) for w in row[name].split()]) for name in row.index.to_numpy()}

def get_contraction_map():
    contraction_df = pd.read_csv("./contractions.csv")
    c_map = {cont:expand for cont, expand in 
            zip(contraction_df['Contraction'], contraction_df['Expanded_Word'])}

    return c_map

def get_default_stopword_strategies():

    nltk.download('stopwords')
    nltk_stopwords= list(nltk.corpus.stopwords.words("english"))

    from spacy.lang.en.stop_words import STOP_WORDS
    spacy_default_stop = list(STOP_WORDS)

    nlp = spacy.load("en_core_web_lg")
    spacy_model_stop = [w.text for w in nlp.vocab if w.is_stop]

    all_stop = nltk_stopwords + spacy_default_stop + spacy_model_stop

    stop_strategies = {"snltk":nltk_stopwords, 
                       "ssd":spacy_default_stop, 
                       "ssm":spacy_model_stop, 
                       "sa":all_stop}
    return stop_strategies

def get_default_remove_strategies():
    remove_strategies = {"text_clean_lp1":"["+string.punctuation.replace("@","")+"]", 
                         "text_clean_lp2":"["+string.punctuation+"]", 
                         "text_clean_lpn":"["+string.punctuation + "0123456789"+"]", 
                         "text_clean_lpa":"[^a-zA-Z]"}
    return remove_strategies


############################## Removing Stopwords
def remove_all_stopwords(df, text_column, custom_stop=None, nltk_stop=True, spacy_default_stop=True, 
                         spacy_model_stop=True, all_stop=True):
    
    stop_df = df.copy(deep=True)
    
    if nltk_stop:
        nltk.download('stopwords')
        nltk_stopwords= nltk.corpus.stopwords.words("english")
        
        stop_df = remove_stopwords(df=stop_df, 
                                   text_column=text_column, 
                                   stopwords=nltk_stopwords, 
                                   stopword_type="snltk") #nltk stopwords
    else:
        nltk_stopwords=[]
    
    if spacy_default_stop:
        from spacy.lang.en.stop_words import STOP_WORDS
        spacy_default_stop=STOP_WORDS
        stop_df = remove_stopwords(df=stop_df, 
                                   text_column=text_column, 
                                   stopwords=spacy_default_stop, 
                                   stopword_type="ssd") #spacy default
    else:
        spacy_default_stop = []
    
    if spacy_model_stop:
        nlp = spacy.load("en_core_web_lg")
        spacy_model_stop = [w.text for w in nlp.vocab if w.is_stop]
        stop_df = remove_stopwords(df=stop_df, 
                                   text_column=text_column, 
                                   stopwords=spacy_model_stop, 
                                   stopword_type="spm") #spacy model
    else:
        spacy_model_stop=[]
    
    if custom_stop is not None:
        stop_df = remove_stopwords(df=stop_df, 
                                   text_column=text_column, 
                                   stopwords=custom_stop, 
                                   stopword_type="scust") #custom 
    else:
        custom_stop=[]
    
    if all_stop:
        all_stopwords = list(spacy_default_stop) + list(spacy_model_stop) + list(nltk_stopwords) + custom_stop
    
        stop_df = remove_stopwords(df=stop_df, 
                                    text_column=text_column, 
                                    stopwords=all_stopwords, 
                                    stopword_type="sall") #all
    
    return stop_df


############################## End Initial Preprocessing ##############################


############################## EDA ##############################
def get_word_freq_dict(df, text_column):
    
    dataframe = df.copy(deep=True)
    dataframe[text_column]=dataframe[text_column].astype(str)
    
    nlp = spacy.load("en_core_web_lg")
    
    string_arrays = dataframe[text_column].tolist()
    all_words = [t.text for txt in string_arrays for t in nlp(txt)]
    
    return nltk.probability.FreqDist(all_words)

def get_class_freq_dicts(df, text_column, class_column="label"):
    
    dataframe = df.copy(deep=True)
    
    categories = df[class_column].unique() 
    
    dfs = {f"{class_column}_{category}":dataframe.loc[dataframe[class_column]==category,:] for category in categories}
    dfs["all"] = df.copy(deep=True)
    
    freq_dicts = {key:get_word_freq_dict(df=dataframe, text_column=text_column) for key, dataframe in dfs.items()}
    return freq_dicts

def get_all_freq_dicts(df, text_columns, save_path="./word_counts/", file_base_name="word_counts"):
    
    data = df.copy(deep=True)
    
    os.makedirs(save_path, exist_ok=True)
    save_name = get_gs_save_name(file_base_name)
    full_save_path = os.path.join(save_path, save_name)
    
    all_freq_dicts = {col:get_class_freq_dicts(df=data, text_column=col) for col in text_columns}
    
    with open(full_save_path, 'wb') as file:
        pickle.dump(all_freq_dicts, file)
        
    return all_freq_dicts
############################## END EDA ##############################

############################## Modeling ##############################

def get_baseline_model_results(df, text_columns, target_column, estimator, metrics, n_cv_splits=10, random_state=42, 
                               shuffle=True, n_jobs=-1, return_estimator=True):
    
    model_df = df.copy(deep=True)
    
    cv = KFold(n_splits=n_cv_splits, shuffle=shuffle, random_state=random_state)
    y = model_df[target_column].to_numpy()
    

    all_results = {}
    for column in text_columns:
    
        X = model_df.loc[:, column].copy(deep=True).astype(str).to_numpy().ravel()
            
        all_results[column] = cross_validate(X=X, y=y, 
                                             estimator=estimator, 
                                             return_estimator=return_estimator,
                                             cv=cv, 
                                             n_jobs=n_jobs, 
                                             return_train_score=True, 
                                             error_score="raise", 
                                             scoring=metrics)
    return all_results


def create_baseline_performance_df(baseline_results, metric="test_f1", smaller_is_better=False):

    preprocess_types = baseline_results.keys()
    dfs = []
    
    funcs = [('min',np.min), ('mean',np.mean), ('max',np.max)]
    
    for preprocess in preprocess_types:
        
        cols = [key for key in baseline_results[preprocess].keys() if key.startswith("train") or key.startswith("test")]
        summary_df = pd.DataFrame({f"{metric}_{agg}":[func(baseline_results[preprocess][metric])] 
                                   for metric in cols for agg, func in funcs})
        
        metric_cols = summary_df.columns.tolist()
        summary_df["preprocess"]=preprocess
        summary_df = summary_df.loc[:,["preprocess"]+metric_cols]
        dfs.append(summary_df)
          
    result_df = pd.concat(objs=dfs).sort_values(by=f"{metric}_mean", 
                                                ascending=smaller_is_better).reset_index(drop=True)
    
    return result_df

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


def run_gridsearch(X, y, estimator, param_grid, cv=None, n_jobs=-1, scoring=None, refit=None, verbose=3, error_score="raise", 
                   return_train_score=True, save=True, save_name=None, base_save_path="./models/", folds=5, shuffle=True, random_state=42):
    

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