# -*- coding: utf-8 -*-

"""
Created on Thu Nov  8 09:20:50 2018

@author: aeppler
"""



import sys
import cv2
import numpy as np

def caption(name, displayBuf):
    
    font = cv2.FONT_HERSHEY_PLAIN
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
    cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    if cap.isOpened():
        windowName = "Default"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Canny Edge Detection")
        showWindow=1  # Show all stages
        

        
        ret_val, frame = cap.read()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        

        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read();
            
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            
            cv2.imshow('frame',fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            
              
            

       
    
    

            if showWindow==1: # Show Camera Frame
                displayBuf = frame
                
                
            elif showWindow == 2:
                #matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = frame
                

            
            caption("press 1, 2, 3, 4, 5, 6, 7, 8 or 9 for frames; s for screenshot", displayBuf)
                

            cv2.imshow(windowName,displayBuf)
            key=cv2.waitKey(10)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key==49: # 1 key, show frame
                cv2.setWindowTitle(windowName,"Camera Feed")
                showWindow=1
            elif key==50: # 2 key
                cv2.setWindowTitle(windowName,"ITK Logo ")
                showWindow=2
                
        
        cap.release()
        cv2.destroyAllWindows()

              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()