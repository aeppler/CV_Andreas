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
            
          
            
            A = cv2.imread('apple.jpg')
            B = cv2.imread('orange.jpg')

            # generate Gaussian pyramid for A
            G = A.copy()
            gpA = [G]
            for i in xrange(6):
                G = cv2.pyrDown(G)
                gpA.append(G)

            # generate Gaussian pyramid for B
            G = B.copy()
            gpB = [G]
            for i in xrange(6):
                G = cv2.pyrDown(G)
                gpB.append(G)

            # generate Laplacian Pyramid for A
            lpA = [gpA[5]]
            for i in xrange(5,0,-1):
                GE = cv2.pyrUp(gpA[i])
                L = cv2.subtract(gpA[i-1],GE)
                lpA.append(L)

            # generate Laplacian Pyramid for B
            lpB = [gpB[5]]
            for i in xrange(5,0,-1):
                GE = cv2.pyrUp(gpB[i])
                L = cv2.subtract(gpB[i-1],GE)
                lpB.append(L)

            # Now add left and right halves of images in each level
            LS = []
            for la,lb in zip(lpA,lpB):
                rows,cols,dpt = la.shape
                ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
                LS.append(ls)

            # now reconstruct
            ls_ = LS[0]
            for i in xrange(1,6):
                ls_ = cv2.pyrUp(ls_)
                ls_ = cv2.add(ls_, LS[i])

            # image with direct connecting each half
            real = np.hstack((A[:,:cols/2],B[:,cols/2:]))


   
    

            if showWindow==1: # Show Camera Frame
                displayBuf = frame
                caption(" ", displayBuf)
                
            elif showWindow == 2: 
                displayBuf = hsv
                caption("hsv", displayBuf)
                
            elif showWindow == 3: 
                displayBuf = edges
                caption("Canny", displayBuf)
                
    




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




              
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()