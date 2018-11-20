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
    cv2.putText(displayBuf, name, (11,20), font, 1.0, (5,5,5), 4, cv2.LINE_AA)  #Text in schwarz
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
            
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            

            
            laplacian = cv2.Laplacian(hsv,cv2.CV_64F)
            sobelx = cv2.Sobel(hsv,cv2.CV_64F,1,0,ksize=5)
            sobely = cv2.Sobel(hsv,cv2.CV_64F,0,1,ksize=5)
   
            
           
            
            # Output dtype = cv2.CV_8U
            sobelx8u = cv2.Sobel(hsv,cv2.CV_8U,1,0,ksize=5)

            # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
            sobelx64f = cv2.Sobel(hsv,cv2.CV_64F,1,0,ksize=5)
            abs_sobel64f = np.absolute(sobelx64f)
            sobel_8u = np.uint8(abs_sobel64f)
   
    

            if showWindow==1: # Show Camera Frame
                displayBuf = hsv
                caption("black white", displayBuf)
                
            elif showWindow == 2: 
                displayBuf = laplacian
                caption("laplacian", displayBuf)
                
            elif showWindow == 3: 
                displayBuf = sobelx
                caption("sobelx", displayBuf)
                
            elif showWindow == 4:
                displayBuf = sobely
                caption("sobely", displayBuf)
                
            elif showWindow == 5:
                displayBuf = sobelx8u
                caption("sobelx8u ", displayBuf)
            
            elif showWindow == 6:
                displayBuf = sobelx64f
                caption("sobelx64f ", displayBuf)
            
            elif showWindow == 7:
                displayBuf = sobel_8u
                caption("sobel_8u", displayBuf)






            cv2.imshow(windowName,displayBuf)
            key=cv2.waitKey(10)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key==49: # 1 key,
                cv2.setWindowTitle(windowName,"Camera Feed")
                showWindow=1
            elif key==50: # 2 key,
                cv2.setWindowTitle(windowName," ")
                showWindow=2
            elif key==51: # 3 key,
                cv2.setWindowTitle(windowName," ")
                showWindow=3
            elif key==52: # 4 key
                cv2.setWindowTitle(windowName," ")
                showWindow=4
            elif key==53: # 4 key
                cv2.setWindowTitle(windowName," ")
                showWindow=5
            elif key==54: # 4 key
                cv2.setWindowTitle(windowName," ")
                showWindow=6
            elif key==55: # 4 key
                cv2.setWindowTitle(windowName," ")
                showWindow=7


              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()