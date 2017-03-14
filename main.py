import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
# Importing Project Related files
import image_processing as ip  # For image processing
import calibration as cal  # Calibrating camera, load, save calibration data
import line  # for line searching, plotting and other lane information



# Global matrix and dist variable because of we want to calculate these onetime only.
# Distortion matrix
mtx = None
# distance
dist = None
# Last found lane
last_lane_img = None



# Main function  to calibrate camera based on checkered images
def camera_calib(loc="camera_cal/calibration*.jpg", camera_calib_data_file='cam_calib.p'):
    # Remove Lens Destortion 1 time per camera only no need to do again and again
    calib_images = glob.glob(loc)
    # get object and image points by calibration images
    objpoints, imgpoints = cal.calibrate_using_checkerd_images(calib_images)
    index = 0
    for cimg in calib_images:
        img = plt.imread(cimg)
        img_size = (img.shape[1], img.shape[0])
        if (len(objpoints) > 0 and len(imgpoints) > 0):
            cal.save_calibration(objpoints, imgpoints, img_size, camera_calib_data_file)
            # Calibrate Camera
            uimg = cal.calibrate_camera(img, objpoints, imgpoints)
            # plt.imshow(uming)
            plt.imsave('output_images/output_camera_calib_{}'.format(index), uimg)
            index += 1
            # plt.show()


# Main Pipeline to process single image/frames in a video
# Set for HD Quality video
def img_pipeline(img, display_step = False):
    global mtx, dist
    global last_lane_img
    # Load Calibration data
    if mtx == None or dist == None:
        # Default calibration file name is 'cam_calib.p'
        mtx, dist = cal.read_calibration()

    if display_step:
        plt.title('Original Image')
        plt.imshow(img)
        plt.show()

    # Remove Camera Lens Distortion
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    if display_step:
        plt.title('Undistorted Image')
        plt.imshow(undist_img)
        plt.show()

    # Apply Color transform and calculating gradients
    thresh_img = ip.apply_thresholds(undist_img, 3, display_step)
    if display_step:
        plt.title('Threshold Image')
        plt.imshow(thresh_img, cmap='gray')
        plt.show()

    # Applying perspective Transform
    wrapped_img, m, minv = ip.wrap(thresh_img)
    if display_step:
        plt.title('Wrapped Image')
        plt.imshow(wrapped_img, cmap='gray')
        plt.show()

    # Detecting Lane Pixels and finding the lane boundaries
    leftx, lefty, rightx, righty = ip.histogram_pixels_v3(wrapped_img,display_step)  # , horizontal_offset=40)

    fit_found = False
    # If left and right lane found
    if len(leftx) > 1 and len(rightx) > 1:
        fit_found = True
    if fit_found:
        try:
            # getting the relative vehicle position, curvature using polynomial
            a, b, c, lx, ly, rx, ry, curvature = line.fit_lanes(wrapped_img)
            # draw polygon mesh on top of the actual image, and project in the real camera view
            poly_img = line.draw_poly(img, thresh_img, a, b, c, lx, ly, rx, ry, minv, curvature)
            # save current frame for future if miss
            last_lane_img = poly_img
            return poly_img
        except:
            return cv2.putText(last_lane_img, '*', (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        # Use last lane
        return cv2.putText(last_lane_img, '*', (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)



def __main__():
    # You can use wildcard entry like ./test_images/test*.jpg
    # to load all jpg images start with word test in test_images folder
    file_s_name = './test_images/test1.jpg'
    files = glob.glob(file_s_name)
    imgs = []
    for file in files:
        imgs.append(plt.imread(file))

    # Remove Camera Distortion
    # Uncomment following line if you want to recompute camera lens destortion
    # camera_calib()

    # This is for testing multiple image
    for img in imgs:
        img = img_pipeline(img, False)
        if type(img) is not None:
            plt.title("Final Image")
            plt.imshow(img, cmap='gray')
            plt.show()

    # Video Processing -- Uncomment following line to process this on video
    #video_processing("project_video.mp4","output-3-project-video.mp4")


def video_processing(file_name, output_file_name=''):
    if output_file_name is '':
        output_file_name = '1_output_' + file_name

    white_output = output_file_name  # 'challenge_video_output.mp4'
    clip1 = VideoFileClip(file_name)  # "challenge_video.mp4")
    white_clip = clip1.fl_image(img_pipeline)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


# Entry of the program
print('Advanced--Lane--Detection--Started')
__main__()
print("Advanced--Lane--Detection--Complete")
