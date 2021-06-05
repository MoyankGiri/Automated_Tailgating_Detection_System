from TailgatingTracking.CentroidTrackingAlgorithm import CentroidTrackingAlgorithm
from TailgatingTracking.TrackableObject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("--prototxt", required=True,help="path to Caffe prototxt file")
ap.add_argument("--caffemodel", required=True,help="path to Caffe .caffemodel file")
ap.add_argument("--input", type=str,help="path to input video file")
ap.add_argument("--output", type=str,help="path to output video file")
ap.add_argument("-cl", "--confidence", type=float, default=0.4,help="minimum probability to remove weak detections")
ap.add_argument("-sf", "--skip-frames", type=int, default=30,help="number of frames skipped")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["caffemodel"])

if not args.get("input", False):
	print("starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("opening video file...")
	vs = cv2.VideoCapture(args["input"])

videowriter = None

WidthOfVideo = None
HeightOfVideo = None

CentroidTracker = CentroidTrackingAlgorithm(MaxFramesAfterDisappeared=40, MaxDistance=50)
trackers = []
trackableObjects = {}


totalFrames = 0
totalCount = 0

fps = FPS().start()

while True:

	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, width=1000)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if WidthOfVideo is None or HeightOfVideo is None:
		(HeightOfVideo, WidthOfVideo) = frame.shape[:2]

	if args["output"] is not None and videowriter is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		videowriter = cv2.VideoWriter(args["output"], fourcc, 30,
			(WidthOfVideo, HeightOfVideo), True)

	status = "Waiting"
	rects = []

	if totalFrames % args["skip_frames"] == 0:

		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (WidthOfVideo, HeightOfVideo), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > args["confidence"]:

				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue

				box = detections[0, 0, i, 3:7] * np.array([WidthOfVideo, HeightOfVideo, WidthOfVideo, HeightOfVideo])
				(startX, startY, endX, endY) = box.astype("int")

				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				trackers.append(tracker)

	else:
		for tracker in trackers:

			status = "Tracking"

			tracker.update(rgb)
			pos = tracker.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY))

	cv2.line(frame, (0, HeightOfVideo // 2), (WidthOfVideo, HeightOfVideo // 2), (0, 255, 255), 2)

	objects = CentroidTracker.UpdateObjectsInFrame(rects)

	for (objectID, centroid) in objects.items():

		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:

				if direction > 0 and centroid[1] > HeightOfVideo // 2:
					totalCount += 1
					to.counted = True

		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	info = [
		("Count", totalCount),
		("Status", status),
	]

	for (i, k) in enumerate(info):
		text = "{}".format(k)
		cv2.putText(frame, text, (10, HeightOfVideo - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	if videowriter is not None:
		videowriter.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	totalFrames += 1
	fps.update()

fps.stop()

if videowriter is not None:
	videowriter.release()

if not args.get("input", False):
	vs.stop()

else:
	vs.release()

cv2.destroyAllWindows()