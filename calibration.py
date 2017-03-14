import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np


def save_calibration(objpoints, imgpoints, img_size, save_calib_file_name='cam_calib.p'):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(save_calib_file_name, "wb"))


def read_calibration(calib_file_name='cam_calib.p'):
    with open(calib_file_name, mode='rb') as f:
        camera_calib = pickle.load(f)
    mtx = camera_calib["mtx"]
    dist = camera_calib["dist"]
    return mtx, dist


def calibrate_camera(img, objpoints, imgpoints, show_img=False):
    # get the size of the camera, HxW
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Perform Undistortion
    undistored_img = cv2.undistort(img, mtx, dist, None, mtx)
    # cv2.imwrite('calibration_wide/test_undist.jpg',dst)

    # Visualize undistortion
    if (show_img == True):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistored_img)
        ax2.set_title('Undistorted Image', fontsize=30)
        f.show()
    return undistored_img


def calibrate_using_checkerd_images(images, nx=9, ny=6, draw_markers=False):
    imgpoints = []
    objpoints = []
    for img_ in images:
        # Read in image
        img = cv2.imread(img_)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates
        # Find the chessboard corners
        # Parameters: (image, chessboard dims, param for any flags)
        # chessboard dims = inside corners, not squares.
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            # Fill image point and object point arrays
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
        if draw_markers == True:
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            plt.show()
    return objpoints, imgpoints


# Perspective transform
def perp_wrap(img, src=None, dst=None):
    imshape = img.shape
    if src is None:
        src = np.float32(
            [[0, 720],
             [550, 470],
             [700, 470],
             [1000, 720]])
    if dst is None:
        dst = np.float32(
            [[200, 720],
             [200, 0],
             [1000, 0],
             [1000, 720]])

    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    plt.imshow(warped, cmap="gray")
    plt.show()


def perp_unwrap(img, src=None, dst=None):
    imshape = img.shape
    if src is None:
        src = np.float32(
            [[0, 720],
             [550, 470],
             [700, 470],
             [1000, 720]])
    if dst is None:
        dst = np.float32(
            [[200, 720],
             [200, 0],
             [1000, 0],
             [1000, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    plt.imshow(warped, cmap="gray")
    plt.show()
