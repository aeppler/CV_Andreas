#!/usr/bin/env python
# MIT License

# Copyright (c) 2017 Jetsonhacks

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import cv2
import numpy as np

def caption(name, displayBuf):
    
    font = cv2.FONT_HERSHEY_PLAIN
    
    cv2.putText(displayBuf, name, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)  #Text in schwarz
    cv2.putText(displayBuf, name, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA) #Text in weiss
            


def read_cam():
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use the Jetson onboard camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        windowName = "Default"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Canny Edge Detection")
        showWindow=1  # Show all stages


        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read();
            

            
            #ret,th1 = cv2.threshold(hsv,127,255,cv2.THRESH_BINARY)
            
     
            
            
            kernel = np.ones((5,5),np.float32)/25
            dst = cv2.filter2D(frame,-1,kernel)
   
    
            blur = cv2.blur(frame,(5,5))
    
            gaussianblur = cv2.GaussianBlur(frame,(5,5),0)

            medianblur = cv2.medianBlur(frame,5)
            
            bilateralblur = cv2.bilateralFilter(frame,9,75,75)
            
            

            if showWindow==1:
                displayBuf = frame
                caption("black white", displayBuf)
                
            elif showWindow == 2:
                displayBuf = dst
                caption("convolution", displayBuf)
                
            elif showWindow == 3: 
                displayBuf = blur
                caption("averaging", displayBuf)
                
            elif showWindow == 4: 
                displayBuf = gaussianblur
                caption("gaussian blur", displayBuf)
                
            elif showWindow == 5: 
                displayBuf = medianblur
                caption("median blur", displayBuf)

            elif showWindow == 6: 
                displayBuf = bilateralblur
                caption("bilateral blur", displayBuf)

                



            cv2.imshow(windowName,displayBuf)
            key=cv2.waitKey(10)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key==49: # 1 key, show frame
                cv2.setWindowTitle(windowName,"Camera Feed")
                showWindow=1
            elif key==50: # 2 key
                cv2.setWindowTitle(windowName,"Black White")
                showWindow=2
            elif key==51: # 3 key
                cv2.setWindowTitle(windowName,"Black White")
                showWindow=3
            elif key==52: # 4 key
                cv2.setWindowTitle(windowName,"Black White")
                showWindow=4
            elif key==53: # 5 key
                cv2.setWindowTitle(windowName,"Black White")
                showWindow=5
            elif key==54: # 6 key
                cv2.setWindowTitle(windowName,"Black White")
                showWindow=6


              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()