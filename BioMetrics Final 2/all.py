import cv2
import numpy as np
import math

# Load the Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
hand_cascade = cv2.CascadeClassifier("aGest.xml")
# Keep track of the last n_positions frames
n_positions = 5
left_shoulder_positions = [0] * n_positions
right_shoulder_positions = [0] * n_positions


contour_history = []

# just for detecting faces
def detect_faces(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    return faces


#for detecting skin colour to detect hands


def detect_shrug(contours):
    shrug_threshold = 5  # Adjust this value based on your requirement
    contour_moments = [cv2.moments(cnt) for cnt in contours]
    contour_centroids = [((m["m10"] / m["m00"]) if m["m00"] != 0 else 0,
                          (m["m01"] / m["m00"]) if m["m00"] != 0 else 0) for m in contour_moments]

    if len(contour_centroids) > 1:
        # Calculate the vertical distance between the centroids
        y_distance = abs(contour_centroids[0][1] - contour_centroids[1][1])

        # Check if the vertical distance between centroids is greater than the threshold
        if y_distance > shrug_threshold:
            cv2.putText(frame, "shrug detected", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return True

    return False

## using canny edge detection and contours to find shoulder
import numpy as np

import numpy as np

def draw_lines_on_edges(frame, img_gray, faces, line_thickness=5):
    edges = cv2.Canny(img_gray, 50, 150)
    shrug_threshold = 5

    for (x, y, w, h) in faces:
        # Modify these values to adjust the breadth of the ROI
        breadth_factor = 2
        start_x = x - int(breadth_factor * w) // 2
        end_x = x + int(breadth_factor * w)

        shoulder_roi = edges[y + h:y + int(1.5 * h), start_x:end_x]
        shoulder_roi_colored = cv2.cvtColor(shoulder_roi, cv2.COLOR_GRAY2BGR)

        # Finding the contours on the detected edges
        contours, _ = cv2.findContours(shoulder_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

        # Find the two largest contours
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Check if the shoulder lines go up significantly
        shrug_detected = False
        for cnt in sorted_contours:
            for point in cnt:
                if point[0][1] < y + h * shrug_threshold:
                    shrug_detected = True
                    break
            if shrug_detected:
                break

        #if shrug_detected:
            #cv2.putText(frame, 'Shrug detected', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw sorted contours on the colored ROI with a different color
        cv2.drawContours(shoulder_roi_colored, sorted_contours, -1, (0, 255, 0), line_thickness)

        # Overlay the colored ROI on the original frame
        frame[y + h:y + int(1.5 * h), start_x:end_x] = cv2.addWeighted(frame[y + h:y + int(1.5 * h), start_x:end_x],
                                                                       0.5, shoulder_roi_colored, 0.5, 0)

    return frame


import cv2

def draw_shoulder_lines(frame, face_rect, both_lines=True, single_line=False, offset=0.2):
    x, y, w, h = face_rect

    left_shoulder_start = right_shoulder_start = left_shoulder_end = right_shoulder_end = None

    if both_lines:
        shoulder_y = y + h + int(offset * h)

        left_shoulder_start = (x, shoulder_y)
        left_shoulder_end = (x - int(0.75 * w), shoulder_y + int(0.25 * h))
        cv2.line(frame, left_shoulder_start, left_shoulder_end, (0, 255, 0), 2)
        cv2.putText(frame, 'left shoulder', (left_shoulder_end[0], left_shoulder_end[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        right_shoulder_start = (x + w, shoulder_y)
        right_shoulder_end = (x + w + int(0.75 * w), shoulder_y + int(0.25 * h))
        cv2.line(frame, right_shoulder_start, right_shoulder_end, (0, 255, 0), 2)
        cv2.putText(frame, 'right shoulder', (right_shoulder_end[0], right_shoulder_end[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 2, 2), 2)

    return left_shoulder_end, right_shoulder_end


###############
def detect_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    return faces[0]


### to detect hands
def skin_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def get_contour_bound_rect(contour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(frame, "Hand detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return x, y, w, h





################
# Open the pre-recorded video file
#cap = cv2.VideoCapture("input1.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and hands in the frame
    faces = detect_faces(frame)
    result = draw_lines_on_edges(frame, img_gray, faces)
    face = detect_face(frame, face_cascade)

    if face is not None:
        x, y, w, h = face
        face_img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 7)

        face_hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        face_mean_color = face_hsv.mean(axis=0).mean(axis=0)

        lower = np.array([face_mean_color[0] - 20, 40, 40], dtype=np.uint8)
        upper = np.array([face_mean_color[0] + 20, 255, 255], dtype=np.uint8)

        mask = skin_mask(frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Sort the contours by area, in descending order
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Get the largest contour
            largest_contour = sorted_contours[0]
            if cv2.contourArea(largest_contour) > 5000:  # filter small contours
                x, y, w, h = get_contour_bound_rect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # Get the second largest contour, if it exists
            if len(sorted_contours) > 1:
                second_largest_contour = sorted_contours[1]
                if cv2.contourArea(second_largest_contour) > 5000:  # filter small contours
                    x, y, w, h = get_contour_bound_rect(second_largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

        skin = cv2.bitwise_and(frame, frame, mask=mask)

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ##########
    for face_rect in faces:
        x, y, w, h = face_rect
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw shoulder lines for the detected face
        draw_shoulder_lines(frame, face_rect, both_lines=True, single_line=False)

        #########


####################


    # Display the frame with the bounding boxes
    #cv2.imshow("Face and Hand Region Tracking", frame)
    #cv2.imshow("Shoulder Detection", result)
    cv2.imshow('Frame', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
