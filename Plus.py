import cv2
import numpy as np

def Plus(red, blue, etal):
    sum = np.zeros_like(etal)
    change = np.zeros_like(etal)
    #changeR = np.zeros_like(etal)
    #chB = np.copy(blue)
    #chR = np.copy(red)
    sum[:, :, 0] = blue
    sum[:, :, 2] = red

    cv2.imwrite("Output/Sum.bmp", sum)

    for (x,y), value in np.ndenumerate(red):
        if (blue[x,y] > 10 and red[x,y] > 10):
            #for i in range(x-1, x+1, 1):
            #    for j in range(y-1, y+1, 1):
            blue[x,y] = 0
            red[x,y] = 0
            #chR[x,y] = 0


        #if (blue[x,y] > 10 and red[x,y] > 10):
        #    chB[x,y] = 0


    #change[:, :, 0] = chB
    #change[:, :, 2] = chR
    change[:, :, 0] = blue
    change[:, :, 2] = red
    cv2.imwrite("Output/Changed.bmp", change)
    return red, blue

