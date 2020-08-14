# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:49:34 2020

@author: HP-Suryansh Bhandari
"""
#Importing required Libraries
import cv2
#import numpy as np

#Using Webcam
vc=cv2.VideoCapture(0)

#set height and width of frame
vc.set(5,800)
vc.set(5,800)

#loading cascades
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
#list of genders and ages
mean_values= (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0,2)','(4,6)','(8,12)','(15,20)','(25,32)','(38,43)','(48,53)','(60,100)']
gender_list=['Male','Female']

#loading pre-trained models
def load_age_and_gender_models():
    age_net=cv2.dnn.readNetFromCaffe('deploy_age.prototxt','age_net.caffemodel')
    gender_net=cv2.dnn.readNetFromCaffe('deploy_gender.prototxt','gender_net.caffemodel')
    return (age_net,gender_net)

def detection(grayscale,img):
    face=cascade_face.detectMultiScale(grayscale,1.1,5)
    if face is ():
        print("No Face Found")
    for(x_face,y_face,w_face,h_face) in face:
        cv2.rectangle(img,(x_face,y_face),(x_face+w_face,y_face+h_face),(255,130,0),2)
        img_grayscale=grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        img_color=img[y_face:y_face+h_face, x_face:x_face+w_face]
        #cv2.imshow(" ",img_color)
        eye=cascade_eye.detectMultiScale(img_grayscale,1.1,10)
        for (x_eye,y_eye,w_eye,h_eye) in eye:
            cv2.rectangle(img_color,(x_eye,y_eye),(x_eye+w_eye,y_eye+h_eye),(0,180,60),2)
        smile=cascade_smile.detectMultiScale(img_grayscale,1.2,20)
        for (x_smile,y_smile,w_smile,h_smile) in smile:
            cv2.rectangle(img_color,(x_smile,y_smile),(x_smile+w_smile,y_smile+h_smile),(255,0,130),2)
    
    return (x_face,y_face,img_color,img)


#if__name=="__main__"
age_net,gender_net=load_age_and_gender_models()
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    _,img=vc.read()
    grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x,y,img_crop,img_whole=detection(grayscale,img)
    blobimg=cv2.dnn.blobFromImage(img_crop,1,(227,227),mean_values,swapRB=False)
    #Predicting gender
    gender_net.setInput(blobimg)
    gender_preds=gender_net.forward()
    gender=gender_list[gender_preds[0].argmax()]
    #Predicting age
    age_net.setInput(blobimg)
    age_preds=age_net.forward()
    age=age_list[age_preds[0].argmax()]
    #printing the text
    overlay_text = "%s %s" % (gender, age)
    cv2.putText(img, overlay_text, (x, y),  font, 1 , (255, 255, 255), 2, cv2.LINE_AA)
    #output
    cv2.imshow('frame',img_whole)
    if (cv2.waitKey(1) & 0xFF == ord('b')) :
        break
    
vc.release()
cv2.destroyAllWindows()

