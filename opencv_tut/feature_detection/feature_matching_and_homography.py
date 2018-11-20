#comparison of orb, sift, surf


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv.imread('box.png',0)          # queryImage
img2 = cv.imread('box_in_scene.png',0) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with ORB and SIFT


kp1_sift, des1_sift = sift.detectAndCompute(img1,None)
kp2_sift, des2_sift = sift.detectAndCompute(img2,None)


kp1_orb, des1_orb = orb.detectAndCompute(img1,None)
kp2_orb, des2_orb = orb.detectAndCompute(img2,None)






#FLANN instead of brute force
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)


matches_orb_flann = flann.knnMatch(des1_sift,des2_sift,k=2)



# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches_orb_flann))]

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches_orb_flann:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1_sift[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2_sift[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None






draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)







img3 = cv.drawMatches(img1,kp1_sift,img2,kp2_sift,good,None,**draw_params)




# Draw first 10 matches.


#cv.imwrite()

cv.imshow('result_surf_flann.jpg',img3)
cv.waitKey(0)
cv.destroyAllWindows()