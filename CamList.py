import numpy as np
import cv2
from timeit import default_timer as timer

def camScan():
    i=0
    list1 = []
    while(i<10):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            list1.append(i)
        i+=1
    return list1











