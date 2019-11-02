#real-time object tracking by estimating homography

import cv2
import numpy as np

# load image
img = cv2.imread('model/the-sun-is-also-a-star.jpg', 1)
# capture real time video
cap = cv2.VideoCapture(0)

# feature detection of model
orb = cv2.ORB_create()
kp_img, des_img = orb.detectAndCompute(img, None)  # argument none=no mask

# featurematching (can use ORB match detector)
# loading the object
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)  # passing setting of the algorithm

while True:
    # read the camera
    _, frame = cap.read()

    # feature detection
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    # findingmatches
    matches = flann.knnMatch(np.asarray(des_img, np.float32), np.asarray(des_frame, np.float32), 2)  # 2

    # neglecting the false matches
    best_matches = []
    for m, n in matches:  # 2 arrays in matches where, m=model and n=object in frame
        # comparing distances
        if m.distance < 0.7 * n.distance:  # ratio test ( the lower the distance between the descriptors the better the match)
            # considering descriptors with shorter distance between them
            best_matches.append(m)

    # Computing Homography
    if len(best_matches) > 10:  # atleast 10 matches - draw homography
        # extracting the position of the points in model by the object m attribute queryIdx in best_matches array
        query_pts = np.float32([kp_img[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_frame[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        # finding homography
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC,
                                          5.0)  # matrix for showing object in perspective

        # through mask we are extracting points into list
        matches_mask = mask.ravel().tolist()

        # perspective transform - passing points of model to adopt in frame
        h, w = img.shape[:-1]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        # drawing line - homography
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        # passing points in integer + join lines + color in bgr + thickness
        cv2.imshow('Object tracking', homography)
    else:
        cv2.imshow('Computing Homography', frame)

    # key event to get out of the loop
    key = cv2.waitKey(1)
    if key == 27:  # escapekey
        break

# release camera
cap.release()
cv2.destroyAllWindows()

