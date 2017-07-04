"""
sumarize.py
"""
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
import csv
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import numpy as np
from numpy import recfromtxt
import os
import re
import sys
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile

Data_file = open('summary_file.txt','r')
fs2 = open('summary.txt','a')

for lines in Data_file:
    if "Number of users collected:" in lines:
        fs2.write(lines)
    elif "Number of messages collected for Sub Graph" in lines:
        fs2.write(lines)
    elif "Number of communities discovered" in lines:
        fs2.write(lines)
    elif "Average number of users per community" in lines:
        fs2.write(lines)
    elif "Number of instances per" in lines:
        fs2.write(lines)
    elif "Positive=" in lines:
        fs2.write(lines)
    elif "Negative=" in lines:
        fs2.write(lines)
    elif "One example" in lines:
        fs2.write(lines)
    elif "Positive_tweet" in lines:
        fs2.write(lines)
    elif "Negative_tweet" in lines:
        fs2.write(lines)
