#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')
#%matplotlib inline
#import warnings
#warnings.filterwarnings('ignore')
#import sklearn.metrics as skm
#import sklearn.model_selection as skms
#import sklearn.preprocessing as skp
#import random
#import librosa, IPython
#import librosa.display as lplt
#seed = 12
#np.random.seed(seed)

# df = pd.read_csv('genre-classification/Data/features_3_sec.csv')
FILE_PATH = 'genre-classification/Data/features_3_sec.csv'
df = pd.read_csv(FILE_PATH)
df.head()

print("Dataset has",df.shape)
print("Count of Positive and Negative samples")
df.label.value_counts().reset_index()