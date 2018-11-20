#comparison of orb, sift, surf


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('box.png',0)          # queryImage
img2 = cv.imread('box_in_scene.png',0) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
sift = cv.xfeatures2d.SIFT_create()
surf = cv.xfeatures2d.SURF_create()

# find the keypoints and descriptors with ORB and SIFT


kp1_sift, des1_sift = sift.detectAndCompute(img1,None)
kp2_sift, des2_sift = sift.detectAndCompute(img2,None)


kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

kp1_surf, des1_surf = surf.detectAndCompute(img1,None)
kp2_surf, des2_surf = surf.detectAndCompute(img2,None)



# create BruteForce Matcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf_sift = cv.BFMatcher()
bf_surf = cv.BFMatcher()


#FLANN instead of brute force
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)


matches_sift_flann = flann.knnMatch(des1_sift,des2_sift,k=2)

# Match descriptors.
matches_orb = bf.match(des1,des2)
matches_sift =  bf_sift.match(des1_sift,des2_sift)
matches_surf =  bf_surf.match(des1_surf,des2_surf)



# Sort them in the order of their distance.
matches_orb = sorted(matches_orb, key = lambda x:x.distance)
matches_sift =  sorted(matches_sift, key = lambda x:x.distance)
matches_surf =  sorted(matches_surf, key = lambda x:x.distance)


# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches_sift_flann))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches_sift_flann):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3_flann = cv.drawMatchesKnn(img1,kp1_sift,img2,kp2_sift,matches_sift_flann,None,**draw_params)




# Draw first 10 matches.
img3_orb = cv.drawMatches(img1,kp1,img2,kp2,matches_orb[:10], None, flags=2)
img3_sift = cv.drawMatches(img1,kp1_sift,img2,kp2_sift,matches_sift[:10], None, flags=2)
img3_surf = cv.drawMatches(img1,kp1_surf,img2,kp2_surf,matches_surf[:10], None, flags=2)


#cv.imwrite()

cv.imshow('result_orb.jpg',img3_orb)
cv.imshow('result_sift.jpg',img3_sift)
cv.imshow('result_surf.jpg',img3_surf)
cv.imshow('result_surf_flann.jpg',img3_flann)
cv.waitKey(0)
cv.destroyAllWindows()