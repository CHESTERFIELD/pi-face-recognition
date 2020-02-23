from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

DATASET_PATH = "dataset"
ENCODINGS_PATH = "encodings.pickle"
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
	"-d", "--dataset",
	help="path to input directory of faces + images",
	default="dataset")
ap.add_argument(
	"-e", "--encodings",
	help="path to serialized db of facial encodings",
	default=ENCODINGS_PATH)
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] Quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] Processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
 
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb)
	
	if len(boxes) == 0:
		print("[WARNING] No faces detected on image \"%s\"" % imagePath)
		continue
	elif len(boxes) > 1:
		print("[WARNING] There must be only one face on image \"%s\"" % imagePath)
		continue
 
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
 
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] Encoded {}/{} faces".format(len(knownEncodings), len(imagePaths)))

# dump the facial encodings + names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] Done")