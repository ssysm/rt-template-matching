import cv2

cv2.startWindowThread()


# Start webcam capture
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = frame.copy()

    # Add helpful text to the frame
    cv2.putText(frame, "Press Enter to capture image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Webcam", frame)

    # Capture image on Enter key press
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        # img = frame
        img_temp = img.copy()
        cv2.imwrite("captured_image.jpg", img)
        break

cap.release()
cv2.destroyAllWindows()

r = cv2.selectROI(img_temp, showCrosshair=True)

if r[2] and r[3]:
    print("Selection is valid")
else:
    print("Selection is invalid, closing...")
    exit(1)

# Crop image
imCrop = img_temp[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
cv2.imwrite("cropped_image.jpg", imCrop)

cv2.destroyAllWindows()
