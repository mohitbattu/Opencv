import os
import numpy as np
import cv2

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('C:/Users/raobk/Desktop/Advanced CV/Opencv/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    
    return faces,gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            #we are avoiding git files or any other mac dot files
            if filename.startswith('.'):
                print("skipping system file")
                continue
            #to abstract its id as 0 and 1
            id=os.path.basename(path)
            #base folder addr is given here
            img_path=os.path.join(path,filename)
            print('img_path',img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
                continue #since we are assuming only the single person images are being fed to classifier
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]#region of interest in gray image
            #we are going to feed face part in our classifer
            faces.append(roi_gray)
            faceID.append(int(id))
            
    return faces,faceID
            
#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
    #Local Binary histogram
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    #this recognizer takes the label as numpy array
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

#Below function draws bounding boxes around detected face in image
def draw_rect(test_img,face):
    #it takes the rectangle cordinates of the face in the image
    #detect Multiscale is going to return the rectangle
    (x,y,w,h)=face
    #creating a bounding box diagonal points are taken
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

#Below function writes name of person for detected label
def put_text(test_img,text,x,y):
    #x,y points are used for marking the position for text
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)

















