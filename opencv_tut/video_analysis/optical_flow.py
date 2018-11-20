# -*- coding: utf-8 -*-

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
        
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7 )
                              
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                         maxLevel = 2,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        #test1
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        
        
        # Take first frame and find corners in it
        # **arg: treat elements as kv-pairs
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        
        

        

        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read();

            #bw frame
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, hsv, p0, None, **lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            
            img = cv2.add(frame,mask)
            
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = hsv.copy()
            p0 = good_new.reshape(-1,1,2)
              
            

       
    
    

            if showWindow==1: # Show Camera Frame
                displayBuf = hsv
                
                
            elif showWindow == 2:
                #matching_img = orb_matching(hsv, itk_screen_bw)                
                displayBuf = hsv
                

            
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