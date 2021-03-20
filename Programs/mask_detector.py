#imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet,maskNet):
    #grab the frame dimensions to construct a blob
    (h,w)=frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 2.0,(224,224),(104.0,177.0,123.0))

    #pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)



    #initialize our list of faces ,their corresponding locations,
    #and the list of predications from our face mask network
    faces =[]
    locs=[]
    preds=[]

    #loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the probability (confidence) assciated with 
        # the detection
        confidence = detections[0,0,i,2]

        #filter out weak detections to ensure
        #confidence is greater than threshold
        if confidence >0.5:
            #compute the (x, y) -coordinates of the bounding box
            #for the object
            box = detections[0,0,i, 3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            #ensure to bounding boxes fall within the dimensions of
            #the frame
            (startX, startY)=(max(0, startX), max(0,startY))
            (endX, endY) = (min(w-1,endX), min(h-1, endY))

            #extarct the face ROI convert it from BGR TO RGB Channel
            #resize it to 224X224 and preprocess it 
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX,startY,endX,endY))
                                                                       
           
    #only make predictions if face detected
    if len(faces) >0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size= 32)
    
    #return two tuples of face locations and their corresponding 
    # locations
    return (locs, preds)
#######################################################################################



#load our serialized face detector model from disk
prototxtPath = r"face_detect/deploy.prototxt"
weightsPath = r"face_detect/res10_300x300_ssd_iter_140000.caffemodel"   
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#load trained mask detection model
maskNet = load_model("mask_detect.model")


# initialize the video stream
print("Starting Video Stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(withoutMask,mask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "No Mask" if mask < withoutMask else "Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                #b g r
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)    

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#cleanup of allocated memory
cv2.destroyAllWindows()
vs.stop()









