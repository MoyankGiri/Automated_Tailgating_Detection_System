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







