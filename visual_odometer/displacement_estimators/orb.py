import cv2
import numpy as np

_orb = cv2.ORB.create()
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)

def orb_analyze_image(img, config: dict):
    kp, des = _orb.detectAndCompute(img, None)
    return [kp, des]

def orb_method(beg, end):
    kp0, des0 = beg
    kp1, des1 = end

    if des0 is not None and des1 is not None:
        matches = _matcher.match(des0, des1)

        kp0 = np.array([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)
        kp1 = np.array([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)

        ret, inliers = cv2.estimateTranslation2D(kp0, kp1, method=cv2.RANSAC)

        return ret[0], ret[1], 0, 1, np.count_nonzero(inliers)/len(inliers)

    else:
        return 0, 0, 0, 1, 0

def orb_method_affine(beg, end):
    kp0, des0 = beg
    kp1, des1 = end

    if des0 is not None and des1 is not None:
        matches = _matcher.match(des0, des1)

        kp0 = (
            np.array([kp0[m.queryIdx].pt for m in matches])
            .reshape(-1, 1, 2)
            .astype(np.float32)
        )
        kp1 = (
            np.array([kp1[m.trainIdx].pt for m in matches])
            .reshape(-1, 1, 2)
            .astype(np.float32)
        )

        print(len(kp1), len(matches), len(des1))

        ret, inliers = cv2.estimateAffine2D(kp0, kp1, method=cv2.RANSAC)

        dx = ret[0, 2]
        dy = ret[1, 2]
        dscale = np.linalg.norm(ret[0:2, 0])
        dtheta = np.atan2(ret[1, 0], ret[0, 0])

        return dx, dy, dtheta, dscale, 1 - np.count_nonzero(inliers) / len(inliers)

    else:
        return 0, 0, 0, 1, 0

