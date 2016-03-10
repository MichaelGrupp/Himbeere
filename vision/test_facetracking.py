# test face detection with OpenCV's face cascade
# video capture with RPi camera module

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2

# import the face cascade
cascPath = '/home/pi/opencv/data/haarcascades/'
faceCascade = cv2.CascadeClassifier(cascPath + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cascPath + 'haarcascade_eye.xml')

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

lost = True

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if lost==True:
                print('detect')
                faces = faceCascade.detectMultiScale(gray, 1.4, 5, minSize=(50, 50))
                print('done')
                print(len(faces))
        if len(faces)>0:
                if lost==True:
                        for (x,y,w,h) in faces:
                                print('found')
                                track_window = x,y,w,h
                                roi_bgr = image[y:y+h, x:x+w]
                                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                                mask = cv2.inRange(roi_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                                roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0,180])
                                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                                lost = False
                                print('tracking...')
                else:
                        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        dst = cv2.calcBackProject([image_hsv], [0], roi_hist, [0,180], 1)
                        #mean shift tracking
                        found, track_window = cv2.meanShift(dst, track_window, term_crit)
                        x,y,w,h = track_window
                        lost = not found
                #draw face box
                if lost == False:
                        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
                                
        # show the frame
	cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
