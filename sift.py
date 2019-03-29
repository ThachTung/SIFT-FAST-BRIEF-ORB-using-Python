
#FAST,BRIEF vs SIFT
'''
import cv2
import numpy as np

img = cv2.imread("jkrowling_holding_book.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("jkrowling_holding_book.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("jkrowling_holding_book.jpg", cv2.IMREAD_GRAYSCALE)

fast = cv2.FastFeatureDetector_create()
sift = cv2.xfeatures2d.SIFT_create()

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp = star.detect(img,None)
kp, des = brief.compute(img,kp)

keypoints = fast.detect (img,None)
keypoints2, descriptors2 = sift.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(img2, keypoints2, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img3 = cv2.drawKeypoints(img3, kp, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("FAST", img)
cv2.imshow("SIFT",img2)
cv2.imshow("BRIEF",img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#SIFT vs ORB
'''
import cv2
import numpy as np

img = cv2.imread("jkrowling_holding_book.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("jkrowling_holding_book.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create(nfeatures=1500)

keypoints, descriptors = sift.detectAndCompute(img, None)
keypoints2, descriptors2 = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(img2, keypoints2, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("SIFT", img)
cv2.imshow("ORB", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''


#ORB and SIFT matching

import cv2
import numpy as np

img1 = cv2.imread("harrypotter_book.jpg", 1)
img2 = cv2.imread("jkrowling_holding_book.jpg", 1)

#SIFT Detector
'''
sift = cv2.xfeatures2d.SIFT_create()
kp3, des3 = sift.detectAndCompute(img1,None)
kp4, des4 = sift.detectAndCompute(img2,None)

# Brute Force Matching
bf = cv2.BFMatcher()

matches = bf.knnMatch(des3, des4, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

matching_result = cv2.drawMatchesKnn(img1, kp3, img2, kp4, good, None, flags=2)

cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# ORB Detector


orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
