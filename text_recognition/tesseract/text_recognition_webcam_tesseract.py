# -*- coding: utf-8 -*-


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
            
       
def textrecognition(image):
    result = pytesseract.image_to_string(image)
    boxes = pytesseract.image_to_boxes(image)
    print result
    
    str = re.sub('[^0-9a-zA-Z -]+', '', result)
    i=1
    

            
    return result, i, str, boxes


def showBoxes(boxes, displayBuf, result):
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(displayBuf.shape) == 2:
        h, w = displayBuf.shape # assumes bw image
        displayBufBGR = cv2.cvtColor(displayBuf, cv2.COLOR_GRAY2BGR)    
    
        
    else:
        h, w,_ = displayBuf.shape # assumes color image
        displayBufBGR=displayBuf
    
    
    if boxes != 0 and len(boxes) > 16:
        for b in boxes.splitlines():
            b = b.split(' ')
            
            #show rectangles where characters were detected
            displayBufBGR = cv2.rectangle(displayBufBGR, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
            
            cv2.imwrite("last_detection.png", displayBufBGR)
                
            
    
    return displayBufBGR



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
        cv2.setWindowTitle(windowName,"Text Recognition")
        showWindow=4  # Show all stages
        boxes = 0
        edgeThreshold=250

        
        #ret_val, frame = cap.read()
        
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            
            kernel_1 = np.ones((1, 1), np.uint8)
            hsv = cv2.dilate(hsv, kernel_1, iterations=1)
            hsv = cv2.erode(hsv, kernel_1, iterations=1)
                    
            # Apply dilation and erosion to remove some noise
            kernel = np.ones((5,5),np.uint8)
            dilate_gray = cv2.dilate(gray, kernel, iterations=1)
            erode_gray = cv2.erode(gray, kernel, iterations=1)
            closing = cv2.dilate(gray, kernel, iterations=1)
            closing = cv2.erode(closing, kernel, iterations=1)
            
            
            opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_1)
            
            gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
            
            blur=cv2.GaussianBlur(gray,(7,7),1.5) #verschwommenes black-white
            blur2 = cv2.GaussianBlur(gray,(3,3),0)
            
            
            canny=cv2.Canny(blur,0,edgeThreshold) #canny filter
            
            
            
            
            # Output dtype = cv2.CV_8U
            sobelx8u = cv2.Sobel(blur,cv2.CV_8U,3,3,ksize=5)
            #sobelx8u2 = cv2.Sobel(blur2,cv2.CV_8U,1,0,ksize=5)
            sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=5)
            

            # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
         
            laplacian = cv2.Laplacian(blur,cv2.CV_64F)
            

            
            # define range of white color in HSV
            # change it according to your need !
            sensitivity = 15
            lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
            upper_white = np.array([255,sensitivity,255], dtype=np.uint8)
                
                
            
            
            
            # Threshold the HSV image to get only white colors
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            
            
            i=i + 1
            result = ''
            #print i
            
            if i == 30 and showWindow==1:
                [result, i, str, boxes] = textrecognition(gray)
                
            
            if i == 30 and showWindow==2:
                
                [result, i, str, boxes] = textrecognition(laplacian)
                    
                    
            if  i == 30 and showWindow==3:
                #result = pytesseract.image_to_string(gradient)
                [result, i, str, boxes] = textrecognition(gradient)
                
                
            if i == 30 and showWindow==4:
                [result, i, str, boxes] = textrecognition(sobelx)
                
            if i == 30 and showWindow==5:
                [result, i, str, boxes] = textrecognition(hsv)
                
            if i == 30 and showWindow==6:
                [result, i, str, boxes] = textrecognition(canny)
                    
                

            if str!='':
                str_show = str
                prev_str = str
            else:
                str_show = prev_str
       
    
    

            if showWindow==1: 
                displayBuf = frame
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                displayBuf = showBoxes(boxes, displayBuf, result)
                
                
                
                
            elif showWindow == 2:
                              
                displayBuf = laplacian
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #displayBuf = showBoxes(boxes, displayBuf, result)
                
                
            elif showWindow == 3:
                                
                displayBuf = gradient
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                displayBuf = showBoxes(boxes, displayBuf, result)
                
            
            elif showWindow == 4:
                                
                displayBuf = sobelx
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                displayBuf = showBoxes(boxes, displayBuf, result)
                
                
                
            elif showWindow == 5:
                                
                displayBuf = hsv
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                displayBuf = showBoxes(boxes, displayBuf, result)
            
            
            
            elif showWindow == 6:
                displayBuf = canny
                cv2.putText(displayBuf,str_show, (230, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                displayBuf = showBoxes(boxes, displayBuf, result)
                
            
            
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
                
              
            elif key==50: # 2 key
                cv2.setWindowTitle(windowName,"sobelx64f ")
                showWindow=2
                
              
            elif key==51: # 3 key
                cv2.setWindowTitle(windowName,"gradient ")
                showWindow=3
                
                
            elif key==52: # 4 key
                cv2.setWindowTitle(windowName,"dilate ")
                showWindow=4
                
            
            elif key==53: # 5 key
                cv2.setWindowTitle(windowName,"mask ")
                showWindow=5
                
            elif key==54: # 6 key
                cv2.setWindowTitle(windowName,"canny ")
                showWindow=6

            elif key==45: # , lower canny edge threshold with -
                edgeThreshold=max(0,edgeThreshold-1)
                print 'Canny Edge Threshold Maximum: ',edgeThreshold
            elif key==43: # . raise canny edge threshold with +
                edgeThreshold=edgeThreshold+1
              
        
        cap.release()
        cv2.destroyAllWindows()

              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()