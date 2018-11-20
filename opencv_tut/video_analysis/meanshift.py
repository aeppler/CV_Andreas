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
        # setup initial location of window
        r,h,c,w = 250,90,400,125  # simply hardcoded the values
        track_window = (c,r,w,h)
        
        # set up the ROI for tracking
        roi = frame[r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #inRange: discard low light values 
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        
        

        

        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read();
            if ret_val == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                
                # apply meanshift to get the new location
                ret_val, track_window = cv2.meanShift(dst, track_window, term_crit)
                
                # Draw it on image
                x,y,w,h = track_window
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                cv2.imshow('img2',img2)
                
                k = cv2.waitKey(60) & 0xff
                if k == 27:
                    break
                else:
                    cv2.imwrite(chr(k)+".jpg",img2)
                    
            else:
                break

              
            

       
    
    

            if showWindow==1: # Show Camera Frame
                displayBuf = hsv
                
                
            elif showWindow == 2:
                #matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = dst
                

            
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
                
        cv2.destroyAllWindows()
        cap.release()

              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()