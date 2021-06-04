from TailgatingTracking.CentroidTrackingAlgorithm import CentroidTrackingAlgorithm
from TailgatingTracking.TrackableObject import TrackableObject
import dlib
import cv2
import imutils
from imutils.video import VideoStream,FPS
import numpy as np
import argparse
import time

ap = argparse.ArgumentParser(description="Inputs for model and some specifications")

ap.add_argument("--prototxt",required=True,help="Path to .prototxt file")
ap.add_argument("--caffemodel",required=True,help="Path to .caffemodel file")
ap.add_argument("--input",help="Input for model",type=str)
ap.add_argument("--output",help="Targeted Output filename",type=str)
ap.add_argument("--frameskipped","-fs",help="Number of frames skipped between detections",default=20,type=int)  #the lower the value the more the computational power is required
ap.add_argument("--confidencelevel","-cl",help="Minimum probability level to filter weak detections",default=0.4,type=float)  
arguments = vars(ap.parse_args())

# Classes on which the SSD was trained
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

print("Loading the MobileNet SSD")
ModelNetwork = cv2.dnn.readNetFromCaffe(arguments["prototxt"],arguments["caffemodel"])

# Input is either the live feed or t an input video
if not arguments.get("input",False):
    print("Using Live Video Stream....")
    vs = VideoStream(src=0).start()

else:
    print("Using path to video provided...")
    vs = cv2.VideoCapture(arguments["input"])

VideoWriter = None
WidthOfFrame,HeightOfFrame = None,None

CentroidTracker = CentroidTrackingAlgorithm()
CorrelationTrackers = []  # dlib's Correlation tracker is used in order to provide the bounding box coordinates of the detected object
TrackableObjects = {}  # ObjectID => Trackable Object

TotalFrames = 0
TotalEntries = 0

fps = FPS().start() #Used to keep a track of frames per second

while True:

    CapturedFrame = vs.read()
    print(CapturedFrame,CapturedFrame[1])











