# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:14:29 2023

@author: dbda
"""

import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

cent_21 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 21st Century.csv")
cent_21=cent_21.drop_duplicates()
