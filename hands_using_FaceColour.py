import cv2
import numpy as np

def detect_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    return faces[0]

def skin_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def get_contour_bound_rect(contour):
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.putText(Frame, "Hand detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return x, y, w, h

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = detect_face(frame, face_cascade)

        if face is not None:
            x, y, w, h = face
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 7)

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

            cv2.imshow('Frame', frame)
            #cv2.imshow('Mask', mask)
            #cv2.imshow('Skin', skin)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
