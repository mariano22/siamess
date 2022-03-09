from torchvision import transforms
from fastbook import *
import PIL
import cv2

def get_x(fn):
    return PILImage.create(fn)

def get_y(fn): return 'CEDAR-'+fn.name.split('_')[1]

def is_validation(fn): return get_y(fn) <= 11
