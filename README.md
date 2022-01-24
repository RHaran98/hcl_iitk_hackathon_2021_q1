## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install -r requirement.txt
```

## Project structure
1. IDS_test.py is the main executable
2. result.csv contains the output of the model on Q1_IDS_test.csv
3. train.py contains the code for training the model
4. Data files in data directory
5. Model logic in utils/model.py
6. requirements.txt has the libraries needed

## Usage
```bash
# Basic usage: Specify input file (Required)
python IDS_test.py <input_file_path>

# Specify output file (Default "result_test.csv")
python IDS_test.py <input_file_path> -o <output_file_path> --date-col <Name of date column of input file> 

# Specify name of date column (Default "DATETIME")
python IDS_test.py <input_file_path> --date-col <Name of date column of input file>
```

## Explanation
1. On plotting the feature columns as line plot, we saw that either the frequency of the sensors were disturbed or there were spikes in the data
2. We used an event based anomaly detection with two kinds of events: Spikes and Frequency changes. Each event detected will set the state to anomaly for a few timestamps. Any further events will reset this clock.
3. To take of care of the spikes, we computed the normal bounds of the feature and predicted an attack for the following few timestamps. Another event would reset the countdown for this.
4. To take care of frequency disruption, we use MACD, with custom thresholds and window sizes for each feature.

## Parameter estimation
1. Parameter estimation was required for MACD
2. For the window size, we computed rolling averages with varying window sizes, and calculated variance for the deviation of overall mean from the window means. 
3. Plotting the variance vs window size as a graph, the window size is the point where variance stops decreasing appreciably with increase in window size. (Third derivative is zero)
4. For the threshold, we simply fix the window size, then observe the range of values mean - rolling mean takes for the normal and attack case, and pick values accordingly.


## License
[MIT](https://choosealicense.com/licenses/mit/)