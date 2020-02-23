from imutils.video import VideoStream
import argparse
import time
import cv2
import os

DATASET_PATH = "dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
SAMPLES_AMOUNT = 5

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
	"-d", "--dataset",
	help="path to input directory of faces + images",
	default=DATASET_PATH)
ap.add_argument("-c", "--cascade",
	help = "path to where the face cascade resides",
	default=CASCADE_PATH)
ap.add_argument(
    "-s", "--samples",
    type=int,
    help="number of samples to take",
    default=SAMPLES_AMOUNT)
args = vars(ap.parse_args())

# load OpenCV's haar cascades for face detection
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# enter person name and id
name = input("Enter user name:  ")
face_id = input("Enter user id:  ")

# create a subdirectory for samples
os.makedirs(args["dataset"] + "/" + name, exist_ok=True)

input("Look at the camera and press Enter to take samples")
count = 0

while(True):
    # read image from camera and detect faces on it
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, 1.3, 5)
    cv2.imshow("image", frame)

    for (x,y,w,h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("{}/{}/{}.{}.jpg".format(args["dataset"], name, face_id, count), frame[y:y+h,x:x+w])

        print("[INFO] Samples taken: {}/{}".format(count, args["samples"]))
        cv2.imshow("image", frame)

    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
    	break
    elif count >= args["samples"]:
        break

# do a bit of cleanup
print("[INFO] Done")
vs.stop()
cv2.destroyAllWindows()