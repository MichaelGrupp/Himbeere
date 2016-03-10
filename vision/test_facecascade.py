# test face detection with OpenCV's face cascade
# video capture with RPi camera module

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
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
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print('detect')
        faces = faceCascade.detectMultiScale(gray, 1.4, 5, minSize=(50, 50))
        print('done')
        for (x,y,w,h) in faces:
                print('found')
                cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
                # optional: eye detection
                #roi_gray = gray[y:y+h, x:x+w]
                #roi_color = image[y:y+h, x:x+w]
                #eyes = eyeCascade.detectMultiScale(roi_gray)
                #for (ex, ey, ew, eh) in eyes:
                #        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        # show the frame
	cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
