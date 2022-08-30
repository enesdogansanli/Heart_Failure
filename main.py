'''
Heart Failure Prediction Project
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# D:\CODING_AREA\PROJECTS\Heart_Failure_Prediction\Data\heart_failure_clinical_records_dataset.csv

df = pd.read_csv("dataset\heart_failure_clinical_records_dataset.csv")
print(df.head())
