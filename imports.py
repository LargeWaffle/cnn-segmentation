import os
import random
from urllib.request import urlopen

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2
import skimage.io as io
from io import BytesIO

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorflow as tf
from keras.metrics import Precision, Recall, AUC
from keras.preprocessing.image import ImageDataGenerator
