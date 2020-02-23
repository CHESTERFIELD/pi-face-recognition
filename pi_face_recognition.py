from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

THRESHOLD = 0.5

CASCADE_PATH = "haarcascade_frontalface_default.xml"
ENCODINGS_PATH = "encodings.pickle"
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade",
	help = "path to where the face cascade resides",
	default=CASCADE_PATH)
ap.add_argument("-e", "--encodings",
	help="path to serialized db of facial encodings",
	default=ENCODINGS_PATH)
ap.add_argument("-t", "--threshold",
	type=float,
	help="threshold for classification",
	default=THRESHOLD)
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] Loading encodings and face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])
 
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
 
# start the FPS counter
fps = FPS().start()

font = cv2.FONT_HERSHEY_SIMPLEX

face_locations = []
face_encodings = []
names = []

frame_number = 0
process_frame = 10

# loop over frames from the video file stream
while True:
	frame_t = time.perf_counter()
	
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	# frame = imutils.resize(frame, width=500)
	
	# convert the frame to grayscale for face detection
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# skip processing for every specified frame
	frame_number = (frame_number + 1) % process_frame
	if frame_number != process_frame - 1:
		continue
	
	# reset names
	names = []

	# detect faces in the grayscale frame
	boxes = detector.detectMultiScale(
		gray,
		scaleFactor=1.1, 
		minNeighbors=5,
		minSize=(50, 50)
	)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]

	# compute the facial embeddings for each face bounding box
	encoding_t = time.perf_counter()
	face_encodings = face_recognition.face_encodings(frame, face_locations)
	print("[DEBUG] Encoded in {:.4f}".format(time.perf_counter() - encoding_t))

	# loop over the facial embeddings
	for encoding in face_encodings:
		
		classification_t = time.perf_counter()

		# calculate distances to current face encoding
		# less distance is better
		distances = face_recognition.face_distance(data["encodings"], encoding)

		sum_distances = {}
		counts = {}

		# calculate sum of all distances of every face encoding
		for (i, distance) in enumerate(distances):
			name = data["names"][i]
			sum_distances[name] = sum_distances.get(name, 0) + distance
			counts[name] = counts.get(name, 0) + 1

		avr_distances = {}

		# calculate average distance of every person to the current
		for name in sum_distances:
			avr_distances[name] = sum_distances.get(name) / counts.get(name)

		# find a minimal distance and check if it passed the threshold
		distance = min(avr_distances.values())
		if (distance < THRESHOLD):
			prediction = min(avr_distances, key=avr_distances.get)
			name = "{}-{}".format(prediction, round(distance * 100))
		else:
			name = "Unknown"
		
		# update the list of nameshtop
		names.append(name)

		print("[DEBUG] Classified in {:.4f}".format(time.perf_counter() - classification_t))

    # loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(face_locations, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		cv2.putText(frame, name, (left + 6, bottom + 29), font, 1, (255, 255, 255), 1)
 
	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
	# update the FPS counter
	fps.update()

	print("[DEBUG] Processed in {:.4f}".format(time.perf_counter() - frame_t))

# stop the timer and display FPS information
fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()