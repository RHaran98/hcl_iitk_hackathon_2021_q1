import pandas as pd
import numpy as np
from collections import deque

class ThresholdCross:
    def __init__(self, feature, decay=1, rise = 1):
        self.watch_type = "threshold" 
        self.feature = feature
        self.threshold = 0
        self.duration = 0
        self.decay = decay
        self.rise = rise
        self.min = True
    
    def compute_threshold(self, data, minimum=True):
        if minimum:
            self.min = True
            self.threshold = np.min(data)
        else:
            self.min = False
            self.threshold = np.max(data)

    def predict(self, point):
        if self.min:
            if point < self.threshold:
                self.duration = self.duration + self.rise
            else:
                self.duration = max([0,self.duration - self.decay])
        else:
            if point > self.threshold:
                self.duration = self.duration + self.rise
            else:
                self.duration = max([0,self.duration - self.decay])
        return self.duration

class RhythmCross:
    def __init__(self, feature, threshold, decay=1, rise = 1):
        self.watch_type = "rhythm"
        self.feature = feature
        self.threshold = threshold
        self.duration = 0
        self.up_freq = None
        self.down_freq = None
        self.rise = rise
        self.decay = decay
        self.min = True
        self.running_up = 0
        self.running_down = 0
    
    def compute_rhythm(self, data):
        data = data - self.threshold
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        edges = []
#         print(zero_crossings)
#             print(splice)
        for i in range(len(zero_crossings)-1):
            splice = data[zero_crossings[i]:zero_crossings[i+1]]
            if np.mean(splice)>0:
                edges.append(len(splice))
            else:
                edges.append(-len(splice))
#         print(edges)
        edges = np.array(edges)
        self.up_freq = np.abs(np.median(edges[edges>0]))
        self.down_freq = np.abs(np.median(edges[edges<0]))
            
        
    def predict(self, point):
        point = point - self.threshold
#         print("Point {} running_up {}/{} running_down {}/{}".format(point, self.running_up, self.up_freq, self.running_down, self.down_freq))
        if point > 0:
            self.running_down = 0
            self.running_up = self.running_up + 1
        elif point < 0:
            self.running_up = 0 
            self.running_down = self.running_down + 1
        if self.running_up > 1.2 * self.up_freq :
            self.duration = self.duration + self.rise
        elif self.running_down > 1.2* self.down_freq:
            self.duration = self.duration + self.rise
        else:
            self.duration = max([0,self.duration - self.decay])
        return self.duration

class MACDCross:
    def __init__(self, feature,threshold, window_size, decay=1, rise = 1):
        self.watch_type = "macd"
        self.feature = feature
        self.threshold = threshold
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.duration = 0
        self.window_stats = None
        self.rise = rise
        self.decay = decay
    
    def compute_macd(self, data):
        self.start_value = np.mean(data)
        
        
    def predict(self, point):
        self.window.append(point)
        if len(self.window) >= self.window_size/2:
            diff = np.abs(self.start_value - np.mean(self.window))
#             print(diff)
            if diff > self.threshold:
                self.duration = self.duration + self.rise
            else: 
                self.duration = max([0,self.duration - self.decay])

        return self.duration

    
class DataHandler:
    def __init__(self, df):
        self.watchers = {}
        self.watch_types = {
            "threshold": ThresholdCross,
            "rhythm": RhythmCross,
            "macd"  : MACDCross
        }
        self.data = df
    def register_watcher(self, config):
        instance = config.get("watch")
        feature = config.get("feature")
        
        if instance == "threshold":
            watch = ThresholdCross(feature, rise = config.get("rise",5), decay = config.get("decay",1))
            watch.compute_threshold(self.data[feature],minimum=config.get("min",False))
            arr = self.watchers.get(feature,[])
            arr.append(watch)
            self.watchers[feature] = arr
            
        elif instance == "rhythm":
            watch = RhythmCross(feature, config.get("threshold",0.5),rise = config.get("rise",1), decay = config.get("decay",2))
            watch.compute_rhythm(self.data[feature])
            arr = self.watchers.get(feature,[])
            arr.append(watch)
            self.watchers[feature] = arr
        elif instance == "macd":
            watch = MACDCross(feature, config.get("threshold",0.5),config.get("window_size",20),rise = config.get("rise",1), decay = config.get("decay",4))
            watch.compute_macd(self.data[feature])
            arr = self.watchers.get(feature,[])
            arr.append(watch)
            self.watchers[feature] = arr

    def get_duration(self, point):
        durations = []
        guess = 0
        for feature in self.watchers:
            watches = self.watchers[feature]
            for watch in watches:
                duration = watch.predict(point[feature])
                durations.append({"watch":watch.watch_type,"feature":feature,"duration":duration})
                guess = max(guess,duration)
        return durations, guess
    

                


