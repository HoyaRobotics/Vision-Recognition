#andrew Johnston and Ethan ross and Taco
import cv2
import numpy as np
from PIL import Image
#note the libraries that need importing
# Camera 0 is the integrated web cam on my netbook
camera_port = 0
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
#camera = cv2.VideoCapture(camera_port)
# Captures a single image from the camera and returns it in PIL format
#def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
# retval, im = camera.read()
# return im
def nothing(x):
    pass
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
#for i in range(ramp_frames):
# temp = get_image()
print("Taking image...")
# Take the actual image we want to keep
#camera_capture = get_image()
#convert to hsv
#hsv = cv2.cvtColor(camera_capture, cv2.COLOR_BGR2HSV)
#set limits for colour to identify
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
lower_green = np.array([30,200,30])
upper_green = np.array([70,255,70])
upper_white = np.array([125,100,125])
lower_white = np.array([0,0,0])
lower_black = np.array([0,0,0])
upper_black = np.array([255,100,100])
#create a mask based on the colour
#mask = cv2.inRange(hsv, lower_green, upper_green)
#combine so the masked area will be visible
#res = cv2.bitwise_and(camera_capture,camera_capture, mask= mask)
#print(res)
#file locations for the test files
#file = "images/test_image1.png"
#file_hsv = "images/hsvtest_image.png"
#file_res = "images/restest_image.png"
# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
#cv2.imwrite(file, camera_capture)
#cv2.imwrite(file_hsv, hsv)
#cv2.imwrite(file_res,res)
cv2.namedWindow("frame1")
# create trackbars for color change, tuned to pink whiffle ball
# lower
cv2.createTrackbar('HLo','frame1',60,179,nothing)
cv2.createTrackbar('SLo','frame1',54,255,nothing)
cv2.createTrackbar('VLo','frame1',24,255,nothing)
# upper
cv2.createTrackbar('HUp','frame1',75,179,nothing)
cv2.createTrackbar('SUp','frame1',255,255,nothing)
cv2.createTrackbar('VUp','frame1',255,255,nothing)
#img = cv2.rectangle("images/restest_image.png",(0,0),(100,100),(0,255,0),3)
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits2
#del(camera)
#def get_image(image_path):
#    """Get a numpy array of an image so that one can access values[x][y]."""
#    image = Image.open(image_path, 'r')
#    width, height = image.size
#    pixel_values = list(image.getdata())
#    if image.mode == 'RGB':
#        channels = 3
#    elif image.mode == 'L':
#        channels = 1
#    else:
#        print("Unknown mode: %s" % image.mode)
#        return None
#    pixel_values = np.array(pixel_values).reshape((width, height, channels))
#
    #print(pixel_values)

#get_image("images/restest_image.png")
#def colourShift():
#
#    image_path = "images/restest_image.png"
#    """Get a numpy array of an image so that one can access values[x][y]."""
#    image = Image.open(image_path, 'r')
    #imageWrite = Image.open(image_path, 'w')
#    width, height = image.size
#    pixel_values = list(image.getdata())
#    new_pixel_values = []
#    for i in range(len(pixel_values)):
#        #print(pixel_values[i])
#        #tempArray = pixel_values[i][k].split("  ")
#        if pixel_values[i][1] > 15:
#            new_pixel_values.append([255,255,255])
#        else:
#            new_pixel_values.append([0,0,0])
    #print("Unknown mode: %s" % image.mode)
#    new_pixel_values = np.array(new_pixel_values).reshape((height, width, 3))
#    return new_pixel_values

#colourShift()
#firstFrame = None
#def calcPixels(frame):
#    pixelValues = list(frame)
#    for i in range(len(pixelValues)):
#        if pixelValues[i][0][2] > 100 & i < len(pixelValues)-1 & pixelValues[i+1][0][2] <100:
#            for c in range(len(pixelValues)):
#                if i+c < len(pixelValues)-1 & pixelValues[i+c][0][2] >100:
#                    return c
#    return "no values found"
def edge(frame):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(frame, 30, 200)
    return edged
def filterNoise(frame):
    #frame = edge(frame)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    image = frame
    #cv2.imshow("test", frame)
    for c in contours:
        if(cv2.contourArea(c)<500):
            continue
        (x,y,w,h) = cv2.boundingRect(c)
	cv2.drawContours(image, [c], 0, (125,125,125), cv2.cv.CV_FILLED)
	#cv2.fillPoly(image, [c], (125,125,125))
        cv2.rectangle(image,(x,y), (x+w,y+h), (255,255,255), 2)
    #cv2.imshow("cont", image)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, None)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, None)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, None)
    return image
cap = cv2.VideoCapture(0)
cap.set(15, -50)
#cap.set(CV_CAP_PROP_MODE, 1)
while (1):
    _, frame = cap.read()
    #print("cap is open ")
    #cap.open(0, cv.CAP_MODE_BGR)
    #if firstFrame is None:
    #    firstFrame = frame
    hLo = cv2.getTrackbarPos('HLo','frame1')
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
    #cv2.imwrite(file_res,res)
    cv2.imshow("res", res)
    cv2.imshow("frame", frame)
    #cv2.imshow("edged", edge(frame))
    cv2.imshow("filtered", filterNoise(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)))
    #cv2.imshow("colour shifted", colourShift())
    #calcPixels(res)
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
