#!usr/bin/python

'''
face-detection.py: Detect faces in jpg images, using cascades for frontal face feature detection from openCV
'''

__author__ = "Stacy Meichle"
__email__ = "sneelin314@gmail.com"
__status__ = "Prototype"

import cv2

# Create cascade classifier object to search for face
face_cascade=cv2.CascadeClassifier("/home/stacy/py-sandbox/data/haarcascade_frontalface_default.xml")

# Load image via imread method as color image
img=cv2.imread("/home/stacy/py-sandbox/data/stacy.jpg")

# Convert to gray scale image to search (easier to classify- increase in accuracy)
gray_scale_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Search for face in image
faces=face_cascade.detectMultiScale(gray_scale_img,
                                    scaleFactor=1.05,
                                    minNeighbors=5)

# Loop through image and draw rectangle around face
for x, y, w, h in faces:
    img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)

print(type(faces))
print(faces)

resized_img=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow("Gray",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
