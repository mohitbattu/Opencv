import cv2
import os
import numpy as np
import Facerecognition as fr


#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread('C:/Users/raobk/Desktop/Advanced CV/Opencv/test images/0/th (2).jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

# for (x,y,w,h) in faces_detected:
#     cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
# resized_img=cv2.resize(test_img,(1000,700))
# cv2.imshow("face detection turoial",resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

#now we want to call labels for training data from our tester function
# faces,faceID=fr.labels_for_training_data('C:/Users/raobk/Desktop/Advanced CV/Opencv/training images')
# face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/raobk/Desktop/Advanced CV/Opencv/trainingData.yml')
#we ginna save the training process which has been done here so that we will not train again and again
# face_recognizer.save('trainingData.yml')
#the above function could be used for predicting newer images
name={0:"Priyanka",1:"Mohit"}
# face_recognizer.write('trainingData.yml')
#face detected for single face
for face in faces_detected:
    #face cordinates for single image
    (x,y,w,h)=face
    #extracting the regions which are faces
    roi_gray=gray_img[y:y+h,x:x+h]
    #we are using this a sthreshold values
    #where it is not confident it gives some values 
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    #Going to create bounding box around rectangle
    fr.draw_rect(test_img,face)
    #now the above function returns the label 
    #now that label is extracted for the label name
    predicted_name=name[label]
    # if(confidence>37):#If confidence more than 37 then don't print predicted face text on screen
    #     continue
    #the predicted name is the txt label
    fr.put_text(test_img,predicted_name,x,y)
 
#we are resizing our image so it may not fit our screen properly
resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows
    
    
    
    
    