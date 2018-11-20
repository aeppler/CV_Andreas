#!/usr/bin/env python


"""
Created on Thu Nov  8 09:20:50 2018

@author: aeppler
"""

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
#import pytesseract

import cv2

# construct the argument parser and parse the arguments
def read_cam():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        windowName = "Default"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Recognition")
        showWindow=4  # Show all stages
        
        #ret_val, frame = cap.read()
        i=1
        
        
        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                    # This will fail if the user closed the window; Nasties get printed to the console
                    break;
            ret_val, frame = cap.read();
            
            
            
            # load the input image and grab the image dimensions
            #image = cv2.imread(args["image"])
            image = frame
            
            #i=i+1
            
            key=cv2.waitKey(10)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break;
                
            #imshow:
            cv2.imshow(windowName,frame) 
            
    else:
        print "camera open failed"
    
    
if __name__ == '__main__':
    read_cam()