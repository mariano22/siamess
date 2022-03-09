import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


from PIL import Image, ImageDraw, ImageOps
import PIL
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import numpy as np
import pytesseract
import os
import gc
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, Normalizer
import sklearn
from shutil import copyfile
from pyzbar.pyzbar import decode
from math import *
import re
from scipy import ndimage
import multiprocessing
import cv2
import random
from collections import defaultdict
from matplotlib import pyplot as plt
from multiprocessing import Pool
import shutil
import functools
import pickle5 as pickle
from scipy import spatial
from scipy import stats
import tempfile
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import time
import copy
import tqdm
import joblib
from fastbook import *
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
import sklearn.metrics
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, recall_score, precision_score, accuracy_score, f1_score 
