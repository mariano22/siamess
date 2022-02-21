
"""
Sigcomp 2009 Train Dataset - NISDICC
====================================
# All author are 001 002 .. 012 (12 authentic) 021 .. 051 (31 forging)
# All filenames matchNISDICC_REGEX
# All authentic signed 5 originals
# In general all forging signed 5 for each authentic author (there are missing falsifications)
# There are missing falsifications: 1898 files instead of 1920
# The image have different sizes
# Imagemode = 'L' (grayscale)

Sigcomp 2009 Test Dataset - NFI
===============================
# All filenames match NFI_REGEX
# All author are 001 002 .. 100 (many miss). Not all of them are falsificators (falsificator << authors)
# In general all authentic signed 12 originals (some 11 and one 6)
# In general all forging signed 6 for 3/4/2 original authors
# There are missing falsifications: 1564 files instead of 1953
# The image have different sizes
# Imagemode = 'L' (grayscale)

"""
from torchvision import transforms
from fastbook import *
import PIL

NFI_REGEX = r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).'

NISDICC_REGEX = r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.'

def get_x(fn):
    #_,img = cv2.threshold(cv2.GaussianBlur(np.array(img),(1,1),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return PILImageBW.create(fn)

def get_y(fn):
    if is_test(fn):
        return get_y_test(fn)
    else:
        return get_y_train(fn)

def get_y_train(fn):
    return 'NISDCC-'+re.match(NISDICC_REGEX,fn.name).group(2)

def get_y_test(fn):
    return 'NFI-'+re.match(NFI_REGEX,fn.name).group(3)

def get_authentic_signatures(path):
    return list(filter(is_authentic, get_image_files(path)))

def is_authentic(fn):
    return is_authentic_train(fn) or is_authentic_test(fn)

def is_authentic_train(fn):
    m = re.match(NISDICC_REGEX,fn.name)
    if m is None:
        return False
    return m.group(1)==m.group(2)

def is_authentic_test(fn):
    m = re.match(NFI_REGEX,fn.name)
    if m is None:
        return False
    return m.group(1)==m.group(3)

def is_test(fn):
    return 'NFI' in fn.name
