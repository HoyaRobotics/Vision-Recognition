#andrew Johnston and Ethan ross and Nolan Meehan
import cv2
import numpy as np
from PIL import Image
#note the libraries that need importing
# Camera 0 is the plugged in webcam on the jetson.
camera_port = 0
def nothing(x):
    pass
#create a window that will contain the trackbars for tuning.
cv2.namedWindow("frame1")
# create trackbars for color change, tuned to retro-reflective tape with green led ring.
# lower bounds
cv2.createTrackbar('HLo','frame1',60,179,nothing)
cv2.createTrackbar('SLo','frame1',54,255,nothing)
cv2.createTrackbar('VLo','frame1',24,255,nothing)
# upper bounds
cv2.createTrackbar('HUp','frame1',75,179,nothing)
cv2.createTrackbar('SUp','frame1',255,255,nothing)
cv2.createTrackbar('VUp','frame1',255,255,nothing)
#function handles the filtering of the frame,aswell as getting the contours.  
def filterNoise(frame):
    #finds the contours in the image.
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    image = frame
    #checks if the contours are larger than 500px in area, and draws them.
    for c in contours:
        if(cv2.contourArea(c)<500):
            continue
        (x,y,w,h) = cv2.boundingRect(c)
	cv2.drawContours(image, [c], 0, (125,125,125), cv2.cv.CV_FILLED)
	#draws a rectangle around the contour.
        cv2.rectangle(image,(x,y), (x+w,y+h), (255,255,255), 2)
    #gets rid of all small loose bits and rounds them out. Makes it look better and was useful previously. may be removed.
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, None)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, None)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, None)
    return image
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
cap = cv2.VideoCapture(0)
while (1):
    #gets a frame from the camera.
    _, frame = cap.read()
    #sets the bounds for the detection from the trackbars.
    sLo = cv2.getTrackbarPos('SLo','frame1')
    vLo = cv2.getTrackbarPos('VLo','frame1')
    hUp = cv2.getTrackbarPos('HUp','frame1')
    sUp = cv2.getTrackbarPos('SUp','frame1')
    vUp = cv2.getTrackbarPos('VUp','frame1')
    #convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of color in HSV
    lower = np.array([hLo,sLo,vLo])
    upper = np.array([hUp,sUp,vUp])
    #create a mask based on the colour
    mask = cv2.inRange(hsv, lower, upper)
    #combine so the masked area will be visible
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #shows the original, partial filtered, and fully filtered and contoured image.
    cv2.imshow("res", res)
    cv2.imshow("frame", frame)
    cv2.imshow("filtered", filterNoise(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)))
    #creates an escape for the loop using k and esc.
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
