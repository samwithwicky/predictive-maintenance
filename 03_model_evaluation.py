import pandas as pd
import numpy as np
import matplotlib as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("data\raw\train_FD001.txt")
