#object tracking in static frame by estimating homography

import cv2
import numpy as np

img1 = cv2.imread('model/the-sun-is-also-a-star.jpg',1)
img2 = cv2.imread('model/query_image.jpg',1)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# Brute Force Matching

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w = img1.shape[:-1]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, homography)
img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

cv2.imshow('Homography',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()