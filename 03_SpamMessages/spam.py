import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import email


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