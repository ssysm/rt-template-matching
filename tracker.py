import cv2
import numpy as np

# Start capturing video
cap = cv2.VideoCapture(2)

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # Convert bounding boxes to float numpy array
    boxes = np.array(boxes, dtype="float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")


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