import numpy as np
import pandas as pd
import pickle
from datetime import datetime


def save_model(model, save_path='auto_mpg/models/', name=""):
    time_stamp = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    save_file_name = f'{save_path}{time_stamp}-{name}.pickle'
    with open(save_file_name, 'wb') as f:
        pickle.dump(model, f)
        print(f'Saved in {save_file_name}')
        
def get_outlier_info(series):
    
    quant_25 = np.quantile(series, 0.25)
    quant_75 = np.quantile(series, 0.75)

    lower_outlier_gate = quant_25 - ((1.5) * (quant_75 - quant_25))
    upper_outlier_gate = quant_75 + ((1.5) * (quant_75 - quant_25))

    
    return lower_outlier_gate, upper_outlier_gate


def get_outlier_val_counts(series, lower_outlier_gate, upper_outlier_gate):
    
    lower_outliers = series[series < lower_outlier_gate]
    upper_outliers = series[series > upper_outlier_gate]

    return {lower_outlier_gate : [len(lower_outliers), lower_outliers], upper_outlier_gate : [len(upper_outliers), upper_outliers]}

def clean_outliers(df, col_name, lower_cutoff_thresh, upper_cutoff_thresh):
    
    df.loc[df[col_name] > upper_cutoff_thresh, col_name] = upper_cutoff_thresh
    df.loc[df[col_name] < lower_cutoff_thresh, col_name] = lower_cutoff_thresh
    
    return df

def train_val_test_split(Xy, train_ratio=0.7, only_test=False):
    
    np.random.shuffle(Xy)
    total_count = Xy.shape[0]
    train_count = int(train_ratio * total_count)
    val_count = int((total_count - train_count)/2)
    test_count = total_count - (train_count + val_count)
    
    return Xy[:train_count], Xy[train_count:train_count+val_count], Xy[train_count+val_count:]
    