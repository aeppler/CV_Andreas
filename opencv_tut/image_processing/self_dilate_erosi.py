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

def read_cam():
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use the Jetson onboard camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        windowName = "CannyDemo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Canny Edge Detection")
        showWindow=4  # Show all stages
        showHelp = True
        font = cv2.FONT_HERSHEY_PLAIN
        helpText="'Esc' to Quit, '1': normal, '2': erosion , '3': dilation , '4': opening , '5': closing , '6': gradient , '7': tophat , '8': blackhat , '9': all, '0': toggle help"
        helpText1="2"
        edgeThreshold=40
        showFullScreen = False
        blackwhite = False
        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            ret_val, frame = cap.read(); #actual image
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts an image from one color space to another
            
            
            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(frame,kernel,iterations = 1)
            dilation = cv2.dilate(frame,kernel,iterations = 1)
            
            opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
            
            
            gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
            tophat = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)

            
            erosionSW = cv2.erode(hsv,kernel,iterations = 1)
            dilationSW = cv2.dilate(hsv,kernel,iterations = 1)
            openingSW = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)
            closingSW = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
            
            gradientSW = cv2.morphologyEx(hsv, cv2.MORPH_GRADIENT, kernel)
            tophatSW = cv2.morphologyEx(hsv, cv2.MORPH_TOPHAT, kernel)
            blackhatSW = cv2.morphologyEx(hsv, cv2.MORPH_BLACKHAT, kernel)

            
            
            
            if showWindow == 4:  # Need to show the 4 stages
                # Composite the 2x2 window
                # Feed from the camera is RGB, the others gray
                # To composite, convert gray images to color. 
                # All images must be of the same type to display in a window
                frameRs=cv2.resize(frame, (426,240)) #links oben 
                
                
                hsvRs=cv2.resize(hsv,(426,240))#rechts oben
                
                
                erosionRs = cv2.resize(erosion,(426,240))
                dilationRs =cv2.resize(dilation,(426,240))
                
                openingRs = cv2.resize(opening,(426,240))
                closingRs = cv2.resize(closing,(426,240))
                
                gradientRs = cv2.resize(gradient, (426,240))
                tophatRs = cv2.resize(tophat, (426,240))
                blackhatRs = cv2.resize(blackhat, (426,240))
                
                
                erosionSWRs = cv2.resize(erosionSW,(426,240))
                dilationSWRs =cv2.resize(dilationSW,(426,240))
                
                openingSWRs = cv2.resize(openingSW,(426,240))
                closingSWRs =cv2.resize(closingSW,(426,240))
                
                gradientSWRs =cv2.resize(gradientSW,(426,240))
                tophatSWRs =cv2.resize(tophatSW,(426,240))
                blackhatSWRs =cv2.resize(blackhatSW,(426,240))
                
                
                
                key2=cv2.waitKey(10) #-1
                if key2 == 115: 
                    blackwhite = True
                
                if key2 == 100: 
                    blackwhite = False
                
                
                
                if blackwhite == False:
                    frame1 = frameRs
                    frame2 = cv2.cvtColor(hsvRs,cv2.COLOR_GRAY2BGR)
                    frame3 = cv2.cvtColor(erosionSWRs,cv2.COLOR_GRAY2BGR)
                    frame4 = cv2.cvtColor(dilationSWRs,cv2.COLOR_GRAY2BGR)
                    frame5 = cv2.cvtColor(openingSWRs,cv2.COLOR_GRAY2BGR)
                    frame6 = cv2.cvtColor(closingSWRs,cv2.COLOR_GRAY2BGR)
                    frame7 = cv2.cvtColor(gradientSWRs,cv2.COLOR_GRAY2BGR)
                    frame8 = cv2.cvtColor(tophatSWRs,cv2.COLOR_GRAY2BGR)
                    frame9 = cv2.cvtColor(blackhatSWRs,cv2.COLOR_GRAY2BGR)
                    
                    
                else:
                    frame1 = frameRs
                    frame2 = frameRs
                    frame3 = erosionRs
                    frame4 = dilationRs
                    frame5 = openingRs
                    frame6 = closingRs
                    frame7 = gradientRs
                    frame8 = tophatRs
                    frame9 = blackhatRs
                        
                # cv2.cvtColor(hsvRs,cv2.COLOR_GRAY2BGR)
                
                vidBuf  = np.concatenate((frame1, frame2, frame3 ), axis=1) #zusammenführen links oben und recht oben
                vidBuf1 = np.concatenate((frame4, frame5,frame6 ), axis=1)
                vidBuf2 = np.concatenate((frame7, frame8, frame9), axis=1)
                
                vidBuf =  np.concatenate( (vidBuf, vidBuf1, vidBuf2), axis=0) #zusammenführen von oben und unten
                
                
                
                

            if showWindow==1: # Show Camera Frame
                displayBuf = frame 
            elif showWindow == 2: # Show Canny Edge Erosion
                displayBuf = erosionSW
            elif showWindow == 3: # Show Canny Edge dilation
                displayBuf = dilationSW
            elif showWindow == 4: # Show All Stages
                displayBuf = vidBuf
            elif showWindow == 5: # Show All opening
                displayBuf = openingSW
            elif showWindow == 6: # Show All closing
                displayBuf = closingSW
            elif showWindow == 7: # Show All gradient
                displayBuf = gradientSW
            elif showWindow == 8: # Show All topHat
                displayBuf = tophatSW
            elif showWindow == 9: # Show All blackHat
                displayBuf = blackhatSW
                

            if showHelp == True:
                cv2.putText(displayBuf, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)  #Text in schwarz
                cv2.putText(displayBuf, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA) #Text in weiss
            
            cv2.imshow(windowName,displayBuf) #show frame
            key=cv2.waitKey(10) #-1
            
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key==49: # 1 key, show frame
                cv2.setWindowTitle(windowName,"Camera Feed")
                showWindow=1
                cv2.putText(displayBuf, helpText1, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA) #Text in weiss
            
            elif key==50: # 2 key, show Erosion
                cv2.setWindowTitle(windowName,"Erosion")
                showWindow=2
            elif key==51: # 2 key, show Dilation
                cv2.setWindowTitle(windowName,"Dilation")
                showWindow=3
            elif key==52: # 4 key, show Opening
                cv2.setWindowTitle(windowName,"Opening")
                showWindow=5
            elif key==53: # 5 key, show Closing
                cv2.setWindowTitle(windowName,"Closing")
                showWindow=6
            elif key==54: # 5 key, show Closing
                cv2.setWindowTitle(windowName,"Gradient")
                showWindow=7
            elif key==55: # 5 key, show Closing
                cv2.setWindowTitle(windowName,"Top Hat")
                showWindow=8
            elif key==56: # 5 key, show Closing
                cv2.setWindowTitle(windowName,"Black Hat")
                showWindow=9
            
            elif key==57: # 6 key, show All
                cv2.setWindowTitle(windowName,"All")
                showWindow=4
            elif key==48: # 0 key, toggle help
                showHelp = not showHelp
            elif key==44: # , lower canny edge threshold
                edgeThreshold=max(0,edgeThreshold-1)
                print 'Canny Edge Threshold Maximum: ',edgeThreshold
            elif key==46: # . raise canny edge threshold
                edgeThreshold=edgeThreshold+1
                print 'Canny Edge Threshold Maximum: ', edgeThreshold
            elif key==53: # Toggle fullscreen; This is the F3 key on this particular keyboard
                # Toggle full screen mode
                if showFullScreen == False : 
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # set to fullscreen
                else:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL) #disable fullscreen
                showFullScreen = not showFullScreen
            
                
    else:
     print "camera open failed"



if __name__ == '__main__':
    read_cam()