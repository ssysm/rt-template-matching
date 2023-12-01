import cv2
import numpy as np
from util import non_max_suppression


def template_matching(template_path):
    # Load the template image
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    # Start capturing video
    cap = cv2.VideoCapture(2)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

      # Create a list of detected bounding boxes
        boxes = []
        for pt in zip(*loc[::-1]):
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

        # Apply non-maximum suppression
        boxes = non_max_suppression(boxes, 0.4) # Adjust the threshold as needed

        # Draw rectangles around detected templates
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    template_image_path = 'cropped_image.jpg' # Replace with your template image path
    template_matching(template_image_path)
