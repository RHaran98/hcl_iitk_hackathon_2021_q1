# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

####################
# Import libraries #
####################
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Custom import from utils folder
from utils.model import DataHandler


#####################
# Read the files in #
#####################
df1 = pd.read_csv(os.path.join("data","Subdataset1.csv"))
df2 = pd.read_csv(os.path.join("data","Subdataset2.csv"))

# Remove extra space in Subdataset2 column names
df2.columns = [c.strip() for c in df2.columns]
df2["DATETIME"] = pd.to_datetime(df2["DATETIME"],format="%d/%m/%y %H")
df1["DATETIME"] = pd.to_datetime(df1["DATETIME"],format="%d/%m/%y %H")



##################################
# Drop, filter and scale columns #
##################################
def preprocess_df(df, fit_scaler=None):
    drop_cols = ["S_PU1","F_PU3","S_PU3","F_PU5","S_PU5","F_PU9","S_PU9"]
    filter_cols = ["DATETIME","F_PU1","L_T1",
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

    df = df.drop(columns=drop_cols)
    
    if fit_scaler:
        scaler = fit_scaler
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])

    df[feature_cols] = scaler.transform(df[feature_cols])
    df[feature_cols] = df[feature_cols] - 0.5
    df = df[filter_cols]
    
    return df, scaler

df1, scaler = preprocess_df(df1)
df2, scaler = preprocess_df(df2,fit_scaler = scaler)



######################
# Remove Attack data #
######################
mask = lambda df : ((df['DATETIME'] >= "2020-09-13 23:00:00") & (df['DATETIME'] <= "2020-09-16 00:00:00") |
                    (df['DATETIME'] >= "2020-09-26 11:00:00") & (df['DATETIME'] <= "2020-09-27 10:00:00") |
                    (df['DATETIME'] >= "2020-10-09 09:00:00") & (df['DATETIME'] <= "2020-10-11 20:00:00") |
                    (df['DATETIME'] >= "2020-10-29 19:00:00") & (df['DATETIME'] <= "2020-11-02 16:00:00") |
                    (df['DATETIME'] >= "2020-11-26 17:00:00") & (df['DATETIME'] <= "2020-11-29 04:00:00") |
                    (df['DATETIME'] >= "2020-12-06 07:00:00") & (df['DATETIME'] <= "2020-12-10 04:00:00") |
                    (df['DATETIME'] >= "2020-12-14 15:00:00") & (df['DATETIME'] <= "2020-12-19 04:00:00") )
attack_df = df2[mask(df2)]



########################################
# Visualize timeseries of each feature #
########################################
v_lines = [50,74,134,228,288,382]
fig, ax =plt.subplots(len(df1.columns)-1,figsize=(30,45))

for i,col in enumerate(df1.columns[1:]):
    sns.lineplot(x=range(len(attack_df)), y=col,
                data = df1.iloc[:492,:],
                 ax=ax[i])

    sax = sns.lineplot(x=range(len(attack_df)), y=col,
                 data=attack_df,
                 ax=ax[i])
    sax.set_xticks(v_lines) # Add vertical lines to differentiate between attacks

## Use Jupyter or plt.pause if you want to see the plots ##
## https://www.kaggle.com/haranr/iitk-hackathon          ##
# fig.tight_layout()
# fig.show()



#########################
# Parameter tuning code #
#########################
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Repeated code to find optimal window size for each feature for MACD
# Change value of c to run for a particular column

# c=1
# c+=1
def find_windowsize(c=1):
    for i in df1.columns[c:c+1]:
        overall_mean = np.mean(df1[i])
        min_std = 1000
        stds = []
        print(i)
        for window_size in range(2,200):
            stds.append(np.std(moving_average(df1[i],window_size)))
        print(2+np.argmin(stds))
    return stds

## Use Jupyter or plt.pause if you want to see the plots ##
## https://www.kaggle.com/haranr/iitk-hackathon          ##
stds = find_windowsize(1)
# sns.lineplot(x=range(len(stds)),y=stds)

# Repeated code to find optimal threshold per column for MACD
# Uncomment print lines to see threshold bands
def find_threshold(win_size=50):
    for i in df1.columns[1:]:
        overall_mean = np.mean(df1[i])
        diffs = []
        
        avgs = moving_average(df1[i],win_size)
        for j in avgs:
            diffs.append(np.abs(j - overall_mean))
        # print("DF",i,np.max(diffs))
        avgs = moving_average(attack_df[i],win_size)
        for j in avgs:
            diffs.append(np.abs(j - overall_mean))
        # print("Attack",i,np.max(diffs))
        # print()

find_threshold(50)



####################
# Model Definition #
####################
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



#########################
# Evaluate model of df2 #
#########################

# 1. Model predicts how long it expects anomaly to continue
# 2. We assume anomaly is occuring when expected duration is greater than 0
rows = []
for _, df_row in df2.iterrows():
    dur, c = dh.get_duration(df_row) # Predict
    row = {i["watch"]+i["feature"]:i["duration"] for i in dur}
    row["duration"] = c
    rows.append(row)

df_stats = pd.DataFrame(rows) # Make a dataframe of the results

# Get hits and misses on df2
hits = len(set(df_stats[df_stats["duration"] > 0].index).intersection(set(attack_df.index)))
misses = len(set(df_stats[df_stats["duration"]>0].index) - set(attack_df.index))
print("Hits and Misses")
print(hits,"/",len(attack_df), hits/len(attack_df) )
print(misses,"/",len(set(df2.index) - set(attack_df)  ), hits/len(set(df2.index) - set(attack_df)  ) )



################
# Export model #
################

# Model is defined by the parameters, which are replicated in the testing file