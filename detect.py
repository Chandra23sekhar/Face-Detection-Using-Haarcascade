import cv2
import numpy as np

# Import haarcascade classifier
classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

no_of_faces = 0 # stores number of faces

# load the image
image=cv2.imread('./car.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert image to grayscale
    
# detect the faces
faces = classifier.detectMultiScale(gray,1.5,5)

if len(faces) == 0:
    print('No faces detected in the image.')

else:
    for (x,y,w,h) in faces:
        no_of_faces += 1
        # draw_rectangle_around_face     
        cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),3)
        
        # # cropping_face_only
        roi_color=image[y:y+h,x:x+w]
        roi_gray=gray[y:y+h,x:x+w]


    msg = "Number of faces in the image: " + str(no_of_faces) 
    cv2.putText(img=image, text=msg, org=(30, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 160, 250),thickness=2) # adding text to image
    cv2.imshow('Faces Identified',image) # show the output image
    cv2.waitKey(0)
    cv2.destroyAllWindows()