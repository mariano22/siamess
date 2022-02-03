from torchvision import transforms
from fastbook import *
import PIL
import cv2

RESIZE_SIZE=105

def get_x(fn):
    return PILImage.create(fn)

def get_y(fn): return fn.parent.name

def is_validation(fn): return fn.parent.name in ['1','9']