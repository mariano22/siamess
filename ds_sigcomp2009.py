



from torchvision import transforms
from fastbook import *
import PIL
import cv2

def get_x(fn):
    img=PIL.Image.open(fn)
    _,img = cv2.threshold(cv2.GaussianBlur(np.array(img),(1,1),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=PIL.Image.fromarray(img)
    img=PILImageBW(img)
    t = ToTensor()(img).float()/255
    return t

def get_y(fn):
    if is_test(fn):
        return get_y_test(fn)
    else:
        return get_y_train(fn)

def get_y_train(fn):
    return re.match(r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.PNG',fn.name).group(2)

def get_y_test(fn):
    return re.match(r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).',fn.name).group(3)

def is_authentic_train(fn):
    m = re.match(r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.PNG',fn.name)
    return m.group(1)==m.group(2)

def is_authentic_test(fn):
    m = re.match(r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).',fn.name)
    return m.group(1)==m.group(3)

def is_test(fn):
    return 'NFI' in fn.name