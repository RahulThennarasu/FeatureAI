import cv2
import numpy as np
import os

path = 'ImageQuery'
orb = cv2.ORB_create(nfeatures=1000)


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

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findID(img, desList, thres=15):
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

    id = findID(img2, desList)
    if id is not None and id != -1:  # Check if id is not None and not -1
        cv2.putText(imgOriginal, classNames[id], (103, 347), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 128), 1)

    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)






# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# print(len(good))
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# # cv2.imshow('Kp1', imgKp1)
# # cv2.imshow('Kp2', imgKp2)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.waitKey(0) 
