#!/usr/bin/env python

"""
Created on Thu Nov  8 09:20:50 2018

@author: aeppler
"""



import sys
import cv2
import numpy as np
import pytesseract
import re

def caption(name, displayBuf):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(displayBuf, name, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)  #Text in schwarz
    cv2.putText(displayBuf, name, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA) #Text in weiss
            
def orb_matching(img1, img2):
    
    orb = cv2.ORB_create() 
    
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

            
    # create BruteForce Matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    
    # match descriptors
    matches_orb = bf.match(des1,des2)
    
    
    # Sort them from best to worst in the order of their distance
    matches_orb = sorted(matches_orb, key = lambda x:x.distance)
    
    
    # draw best 50 matches
    return cv2.drawMatches(img1,kp1,img2,kp2,matches_orb[:50], None, flags=2)

            




def read_cam():
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use the Jetson onboard camera
    #cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        windowName = "Default"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Canny Edge Detection")
        showWindow=2  # Show all stages
        

        
        ret_val, frame = cap.read()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        i=1
        str = ''
        prev_str = ''
        str_show = ''

        

        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read();
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
            # Apply dilation and erosion to remove some noise
            kernel = np.ones((1,1),np.uint8)
            dilate_hsv = cv2.dilate(hsv, kernel, iterations=1)
            erode_hsv = cv2.erode(hsv, kernel, iterations=1)
            closing = cv2.dilate(hsv, kernel, iterations=1)
            closing = cv2.erode(closing, kernel, iterations=1)
            
            
            gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
            
             
            
            
    
            
            # define range of white color in HSV
            # change it according to your need !
            sensitivity = 15
            lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
            upper_white = np.array([255,sensitivity,255], dtype=np.uint8)
                
                
            
            
            
            # Threshold the HSV image to get only white colors
            #mask = cv2.inRange(hsv, lower_white, upper_white)
            
            
            
            i=i + 1
            result = ''
            if i == 30:
                
                if showWindow==2:
                    result = pytesseract.image_to_string(closing)
                    
                    
                if showWindow==3:
                    result = pytesseract.image_to_string(gradient)
                if showWindow==4:
                    result = pytesseract.image_to_string(dilate_hsv)
                       
                    
                    
                print result
                str = re.sub('[^0-9a-zA-Z -]+', '', result)
                i=1

            if str!='':
                str_show = str
                prev_str = str
            else:
                str_show = prev_str
       
    
    

            if showWindow==1: # Show Camera Frame
                displayBuf = frame
                
                
            elif showWindow == 2:
                #matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = closing
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                
            elif showWindow == 3:
                #matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = gradient
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
            
            elif showWindow == 4:
                #matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = dilate_hsv
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
            
            
            
            #cv2.imshow('frame',frame)
            
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

            
            #caption("press 1, 2, 3, 4, 5, 6, 7, 8 or 9 for frames; s for screenshot", displayBuf)
                

            cv2.imshow(windowName,displayBuf)
            key=cv2.waitKey(10)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key==49: # 1 key, show frame
                cv2.setWindowTitle(windowName,"Camera Feed")
                showWindow=1
                caption("gray", displayBuf)
              
            elif key==50: # 2 key
                cv2.setWindowTitle(windowName,"closing ")
                showWindow=2
                caption("gray", displayBuf)
              
            elif key==51: # 3 key
                cv2.setWindowTitle(windowName,"GAUSS ")
                showWindow=3
                caption("gauss", displayBuf)
                
            elif key==52: # 3 key
                cv2.setWindowTitle(windowName,"dilate ")
                showWindow=4
                caption("dilate", displayBuf)
              
        
        cap.release()
        cv2.destroyAllWindows()

              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()