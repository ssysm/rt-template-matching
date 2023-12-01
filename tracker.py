import cv2
import numpy as np
from util import non_max_suppression

# Start capturing video
cap = cv2.VideoCapture(2)


def template_matching(template_path):
    # Load the template image
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

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

        # Break the loop when 'enter' key is pressed
        # and create and launch trackers
        trackers = []
        if cv2.waitKey(1) & 0xFF == 13:
            for box in boxes:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1])) # (startX, startY, (endX - startX), (endY - startY))
                trackers.append(tracker)
            launch_tracker(trackers)

    # When everything done, release the capture
    cv2.destroyAllWindows()

def launch_tracker(tracker_list):
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update trackers
        timer = cv2.getTickCount()
        for tracker in tracker_list:
            ok, bbox = tracker.update(frame)
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        # Display the resulting frame
        cv2.imshow('Tracker Frame', frame)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    template_image_path = 'cropped_image.jpg' # Replace with your template image path
    template_matching(template_image_path)