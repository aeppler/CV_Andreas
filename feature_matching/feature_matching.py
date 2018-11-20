#!/usr/bin/env python
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
    
    
    # Match descriptors.
    matches_orb = bf.match(des1,des2)
    
    
    # Sort them in the order of their distance.
    matches_orb = sorted(matches_orb, key = lambda x:x.distance)


    # Draw first 10 matches.
    return cv2.drawMatches(img1,kp1,img2,kp2,matches_orb[:10], None, flags=2)

            



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
        
        counter = 0

        

        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read();

            #bw frame
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
            
            #itk screenshot
            itk_screen_c = cv2.imread('itk_logo_screen.png')
            itk_screen_bw = cv2.imread('itk_logo_screen.png',0)
            
            #itk logo
            itk_logo_c= cv2.imread('itk_logo.png')
            itk_logo_bw= cv2.imread('itk_logo.png', 0)
               
            #bellaris screenshot
            bellaris_screen_c = cv2.imread('bellaris_screen.png')
            bellaris_screen_bw = cv2.imread('bellaris_screen.png', 0)
            
            



            # Draw first 10 matches.
            

            #cv2.imshow('itk', itk_img_c)
            #cv2.imshow('harris corners',frame)
            #cv2.imshow('img3 orb',img3_orb)
       
    
    

            if showWindow==1: # Show Camera Frame
                displayBuf = hsv
                
                
            elif showWindow == 2:
                matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = matching_img
                
            elif showWindow == 3:
                matching_img = orb_matching(frame, itk_screen_c)                
                displayBuf = matching_img
                #caption("colored orb matching itk screen", displayBuf)
                
            elif showWindow == 4:
                matching_img = orb_matching(hsv, itk_logo_bw)                
                displayBuf = matching_img
                #caption("black white orb matching itk logo", displayBuf)
                
            elif showWindow == 5:
                matching_img = orb_matching(frame, itk_logo_c)                
                displayBuf = matching_img
                #caption("colored white orb matching itk logo", displayBuf)
            
            elif showWindow == 6:
                matching_img = orb_matching(frame, bellaris_screen_c)                
                displayBuf = matching_img
                #caption("colored white orb matching itk logo", displayBuf)
            
            
            caption("press 1, 2, 3, 4, 5 or 6 for frames; 7 for screenshot", displayBuf)
                

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
            elif key==51: # 3 key
                cv2.setWindowTitle(windowName,"ITK Logo ")
                showWindow=3
            elif key==52: # 4 key
                cv2.setWindowTitle(windowName,"ITK Logo ")
                showWindow=4
            elif key==53: # 5 key
                cv2.setWindowTitle(windowName,"ITK Logo ")
                showWindow=5
            elif key==54: # 6 key
                cv2.setWindowTitle(windowName,"Bellaris ")
                showWindow=6
            elif key==55: # 7 key, screenshot
                cv2.imwrite("frame%d.jpg" % counter, displayBuf)
                cv2.imshow("frame%d.jpg" % counter, displayBuf)
                
                counter +=1


              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()
