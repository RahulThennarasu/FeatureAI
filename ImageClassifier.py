import cv2
import numpy as np
import os
import pyttsx3

path = 'ImageQuery'
orb = cv2.ORB_create(nfeatures=1500)

#### Import Images
images = []
classNames = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    if cl == '.DS_Store':
        continue  # Skip this file
    imgCurrent = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCurrent)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to speak the given text
def speak(text):
    os.system(f'say {text}')

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findID(img, desList, thres=20):
    kp2, des2 = orb.detectAndCompute(img, None)
    if des2 is None:
        print("No descriptors found for query image.")
        return
    
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            if des is None:
                print("Descriptors not found for a reference image.")
                matchList.append(0)
                continue
            
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    # print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal

desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Flip the image horizontally
    imgOriginal = cv2.flip(imgOriginal, 1)

    id = findID(img2, desList)
    if id is not None and id != -1:  # Check if id is not None and not -1
        text_to_speak = classNames[id]  # Get the text to speak
        cv2.putText(imgOriginal, text_to_speak, (103, 347), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 128), 1)
        speak(text_to_speak)  # Speak the text

    cv2.imshow('img2', imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
