

from torchvision import transforms
from fastbook import *
import PIL
import cv2

def get_x(fn):
    return PILImageBW.create(fn)

def get_y(fn): return fn.parent.parent.name+'-'+fn.parent.name

def is_validation(fn): return fn.parent.parent.parent.name=='images_evaluation'