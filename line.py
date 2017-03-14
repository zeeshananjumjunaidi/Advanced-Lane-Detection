import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # x values in windows
        self.windows = np.ones((3, 12)) * -1


left_lane = Line()
right_lane = Line()


def find_nearest(array, value):
    # Function to find the nearest point from array
    if len(array) > 0:
        idx = (np.abs(array - value)).argmin()
        return array[idx]


def find_peaks(image, y_window_top, y_window_bottom, x_left, x_right):
    # Find the historgram from the image inside the window
    histogram = np.sum(image[y_window_top:y_window_bottom, :], axis=0)
    # Find the max from the histogram
    if len(histogram[int(x_left):int(x_right)]) > 0:
        return np.argmax(histogram[int(x_left):int(x_right)]) + x_left
    else:
        return (x_left + x_right) / 2


def sanity_check(lane, curverad, fitx, fit):
    # Sanity check for the lane
    if lane.detected:  # If lane is detected
        # If sanity check passes
        if abs(curverad / lane.radius_of_curvature - 1) < .6:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curverad
            lane.current_fit = fit
        # If sanity check fails use the previous values
        else:
            lane.detected = False
            fitx = lane.allx
    else:
        # If lane was not detected and no curvature is defined
        if lane.radius_of_curvature:
            if abs(curverad / lane.radius_of_curvature - 1) < 1:
                lane.detected = True
                lane.current_fit = fit
                lane.allx = fitx
                lane.bestx = np.mean(fitx)
                lane.radius_of_curvature = curverad
                lane.current_fit = fit
            else:
                lane.detected = False
                fitx = lane.allx
        # If curvature was defined
        else:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curverad
    return fitx


# Sanity check for the direction
def sanity_check_direction(right, right_pre, right_pre2):
    # If the direction is ok then pass
    if abs((right - right_pre) / (right_pre - right_pre2) - 1) < .2:
        return right
    # If not then compute the value from the previous values
    else:
        return right_pre + (right_pre - right_pre2)


# find_lanes function will detect left and right lanes from the warped image.
# 'n' windows will be used to identify peaks of histograms
def find_lanes(n, image, x_window, lanes,
               left_lane_x, left_lane_y, right_lane_x, right_lane_y, window_ind):
    # 'n' windows will be used to identify peaks of histograms
    # Set index1. This is used for placeholder.
    index1 = np.zeros((n + 1, 2))
    index1[0] = [300, 1100]
    index1[1] = [300, 1100]
    # Set the first left and right values
    left, right = (300, 1100)
    # Set the center
    center = 700
    # Set the previous center
    center_pre = center
    # Set the direction
    direction = 0
    for i in range(n - 1):
        # set the window range.
        y_window_top = 720 - 720 / n * (i + 1)
        y_window_bottom = 720 - 720 / n * i
        # If left and right lanes are detected from the previous image
        if (left_lane.detected == False) and (right_lane.detected == False):
            # Find the historgram from the image inside the window
            left = find_peaks(image, y_window_top, y_window_bottom, index1[i + 1, 0] - 200, index1[i + 1, 0] + 200)
            right = find_peaks(image, y_window_top, y_window_bottom, index1[i + 1, 1] - 200, index1[i + 1, 1] + 200)
            # Set the direction
            left = sanity_check_direction(left, index1[i + 1, 0], index1[i, 0])
            right = sanity_check_direction(right, index1[i + 1, 1], index1[i, 1])
            # Set the center
            center_pre = center
            center = (left + right) / 2
            direction = center - center_pre
        # If both lanes were detected in the previous image
        # Set them equal to the previous one
        else:
            left = left_lane.windows[window_ind, i]
            right = right_lane.windows[window_ind, i]
        # Make sure the distance between left and right laens are wide enough
        if abs(left - right) > 600:
            # Append coordinates to the left lane arrays
            left_lane_array = lanes[(lanes[:, 1] >= left - x_window) & (lanes[:, 1] < left + x_window) &
                                    (lanes[:, 0] <= y_window_bottom) & (lanes[:, 0] >= y_window_top)]
            left_lane_x += left_lane_array[:, 1].flatten().tolist()
            left_lane_y += left_lane_array[:, 0].flatten().tolist()
            if not math.isnan(np.mean(left_lane_array[:, 1])):
                left_lane.windows[window_ind, i] = np.mean(left_lane_array[:, 1])
                index1[i + 2, 0] = np.mean(left_lane_array[:, 1])
            else:
                index1[i + 2, 0] = index1[i + 1, 0] + direction
                left_lane.windows[window_ind, i] = index1[i + 2, 0]
            # Append coordinates to the right lane arrays
            right_lane_array = lanes[(lanes[:, 1] >= right - x_window) & (lanes[:, 1] < right + x_window) &
                                     (lanes[:, 0] < y_window_bottom) & (lanes[:, 0] >= y_window_top)]
            right_lane_x += right_lane_array[:, 1].flatten().tolist()
            right_lane_y += right_lane_array[:, 0].flatten().tolist()
            if not math.isnan(np.mean(right_lane_array[:, 1])):
                right_lane.windows[window_ind, i] = np.mean(right_lane_array[:, 1])
                index1[i + 2, 1] = np.mean(right_lane_array[:, 1])
            else:
                index1[i + 2, 1] = index1[i + 1, 1] + direction
                right_lane.windows[window_ind, i] = index1[i + 2, 1]
    return left_lane_x, left_lane_y, right_lane_x, right_lane_y


def find_curvature(yvals, fitx):
    # Define y-value where we want radius of curvature
    # I choose the maximum y-value, from bottom of the image
    y_eval = np.max(yvals)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    fit_cr = np.polyfit(yvals * ym_per_pix, fitx * xm_per_pix, 2)
    curve = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curve


def find_position(pts, image_shape):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = image_shape[1] / 2
    # print(pts.shape)
    try:
        left = np.min(pts[(pts[:, 1] < position) & (pts[:, 0] > 700)][:, 1])
        right = np.max(pts[(pts[:, 1] > position) & (pts[:, 0] > 700)][:, 1])
        center = (left + right) / 2
        # Define conversions in x and y from pixels space to meters
        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
        return (position - center) * xm_per_pix
    except:
        return 0


# Function to find the fitting lines from the warped image
def fit_lanes(image):
    # define y coordinate values for plotting
    yvals = np.linspace(0, 100, num=101) * 7.2  # to cover same y-range as image
    # find the coordinates from the image
    lanes = np.argwhere(image)
    # Coordinates for left lane
    left_lane_x = []
    left_lane_y = []
    # Coordinates for right lane
    right_lane_x = []
    right_lane_y = []
    # Find lanes from three repeated procedures with different window values
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(4, image, 25, lanes,
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 0)
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(6, image, 50, lanes,
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 1)
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(8, image, 75, lanes,
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 2)
    # Find the coefficients of polynomials
    left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]
    # Find curvatures
    left_curverad = find_curvature(yvals, left_fitx)
    right_curverad = find_curvature(yvals, right_fitx)
    # Sanity check for the lanes
    left_fitx = sanity_check(left_lane, left_curverad, left_fitx, left_fit)
    right_fitx = sanity_check(right_lane, right_curverad, right_fitx, right_fit)

    return yvals, left_fitx, right_fitx, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_curverad


# draw poly on an image
def draw_poly(image, warped, yvals, left_fitx, right_fitx,
              left_lane_x, left_lane_y, right_lane_x, right_lane_y, Minv, curvature):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    #pts_center = np.int_([pts_right - pts_left])

    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (30, 180, 0))
    #cv2.polylines(color_warp, pts_center, False, (0, 50, 200), 10, 16)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    # Put text on an image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(result, text, (402, 102), font, 1, (0, 0, 0), 3)
    cv2.putText(result, text, (400, 100), font, 1, (255, 255, 255), 3)
    # Find the position of the car
    pts = np.argwhere(newwarp[:, :, 1])
    position = find_position(pts, image.shape)
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    cv2.putText(result, text, (402, 152), font, 1, (0, 0, 0), 3)
    cv2.putText(result, text, (400, 150), font, 1, (255, 255, 255), 3)
    return result
