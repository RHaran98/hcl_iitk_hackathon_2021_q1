####################
# Import libraries #
####################
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
import argparse

# Custom import from utils folder
from utils.model import DataHandler


##################################
# Drop, filter and scale columns #
##################################
def preprocess_df(df, fit_scaler=None, date_col= "DATETIME"):
    filter_cols = [date_col,"F_PU1","L_T1",
    "F_PU2",
    "F_PU6",
    "F_PU7",
    "F_PU11",
    "S_PU11",
    "P_J280",
    "P_J269",
    "P_J415",
    "P_J302",
    "P_J307",
    "P_J317",
    "P_J14"]

    feature_cols = filter_cols[1:] # IGNORE Datetime column

    
    if fit_scaler:
        scaler = fit_scaler
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])

    df[feature_cols] = scaler.transform(df[feature_cols])
    df[feature_cols] = df[feature_cols] - 0.5
    df = df[filter_cols]
    
    return df, scaler



####################
# Model Definition #
####################
def get_model():
    dh = DataHandler(df1) # Initialize model

    # Add handpicked parameters for macd
    dh.register_watcher({"feature":"F_PU1","watch":"macd","threshold":0.17,"window_size":50})
    dh.register_watcher({"feature":"L_T1","watch":"macd","threshold":0.21,"window_size":46})
    dh.register_watcher({"feature":"F_PU6","watch":"macd","threshold":0.12,"window_size":25})
    dh.register_watcher({"feature":"F_PU7","watch":"macd","threshold":0.16,"window_size":20})
    dh.register_watcher({"feature":"F_PU11","watch":"macd","threshold":0.1,"window_size":25})
    dh.register_watcher({"feature":"S_PU11","watch":"macd","threshold":0.1,"window_size":25})
    dh.register_watcher({"feature":"P_J280","watch":"macd","threshold":0.23,"window_size":50})
    dh.register_watcher({"feature":"P_J269","watch":"macd","threshold":0.17,"window_size":50})
    dh.register_watcher({"feature":"P_J302","watch":"macd","threshold":0.1,"window_size":45})
    dh.register_watcher({"feature":"P_J307","watch":"macd","threshold":0.1,"window_size":45})
    dh.register_watcher({"feature":"P_J317","watch":"macd","threshold":0.1,"window_size":20})
    dh.register_watcher({"feature":"P_J14","watch":"macd","threshold":0.1,"window_size":50})

    # Add handpicked parmeters for threshold (By observing which side spikes occur in, positive or negative)
    dh.register_watcher({"feature":"F_PU1","watch":"threshold","min":True})
    dh.register_watcher({"feature":"F_PU6","watch":"threshold","min":False})
    dh.register_watcher({"feature":"F_PU11","watch":"threshold","min":False,"decay":10})
    dh.register_watcher({"feature":"S_PU11","watch":"threshold","min":False,"decay":10})
    dh.register_watcher({"feature":"P_J280","watch":"threshold","min":False})
    dh.register_watcher({"feature":"P_J269","watch":"threshold","min":False})
    dh.register_watcher({"feature":"P_J415","watch":"threshold","min":False})
    dh.register_watcher({"feature":"P_J302","watch":"threshold","min":False})
    dh.register_watcher({"feature":"P_J307","watch":"threshold","min":False})
    dh.register_watcher({"feature":"P_J317","watch":"threshold","min":False,"decay":10})
    dh.register_watcher({"feature":"P_J14","watch":"threshold","min":False})
    return dh

################
# Testing code #
################
if __name__ == "__main__":
    # Define arguments parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'),help="Name of input file",default="test.csv")
    parser.add_argument('-o', '--output', help="Path of output file",default="result_test.csv")
    parser.add_argument( '--date-col', help="Name of the date column in the input file",default="DATETIME")
    args = parser.parse_args() 

    # Extract commandline arguements
    input_file_name = args.input_file.name
    output_file_name = args.output
    date_col = args.date_col

    # Get scaler and model
    df1 = pd.read_csv(os.path.join("data","Subdataset1.csv"))
    df1, scaler = preprocess_df(df1)
    dh = get_model()
    
    # Read in input file and preprocess
    df = pd.read_csv(input_file_name)
    df, scaler = preprocess_df(df,fit_scaler = scaler, date_col=date_col)

    # Predict
    rows = []
    for _, df_row in df.iterrows():
        dur, c = dh.get_duration(df_row) # Predict
        if c > 0:
            label = "ATTACK"
        else: 
            label = "NORMAL"
        row = {"TIME":df_row[date_col], "LABEL":label}
        rows.append(row)

    df_op = pd.DataFrame(rows) # Make a dataframe of the results
    df_op = df_op[["TIME","LABEL"]]

    # Export results
    df_op.to_csv(output_file_name, index=False)
    print("Exported to {}".format(output_file_name))

    


