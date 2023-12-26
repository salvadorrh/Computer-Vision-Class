# Capture and display images from webcam
# Press 'q' to exit
# Press 'w' to make a prototype from current frame
# Press 'i' to toggle in and out of identify mode

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

im_rows, im_cols =  720, 960
cap = cv2.VideoCapture(0)
cap.set(3,im_cols)
cap.set(4,im_rows)

start = time.time()
count = 0
identify = False
 # L are the frames. N are the number of matches. 
 # K are the keypoints and D are the descriptors.
L, N, K, D = [], [], [], []

# Create ORB instance
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
imgs_to_read = ['img_0.jpg', 'img_1.jpg']

# Function that will get the ransac matches between Frame and Prototype
def get_ransac_matches(keypointsFrame, descriptorsFrame, keypointsProt,
                       descriptorsProt):
    matches = matcher.match(descriptorsFrame,descriptorsProt)
    matches = sorted(matches, key = lambda x:x.distance)
    pts0 = np.zeros((len(matches),2))
    pts1 = np.zeros((len(matches),2))
    for i, m in enumerate(matches[:len(matches)]):
        q = m.queryIdx
        t = m.trainIdx
        pts0[i] = np.asarray(keypointsFrame[q].pt)
        pts1[i] = np.asarray(keypointsProt[t].pt)
        
    H, m = cv2.findHomography(pts0.reshape(-1,1,2), pts1.reshape(-1,1,2), cv2.RANSAC)
    m = np.array(m).reshape(-1)
    m_ret = m.tolist().count(True)  # Get the number of matches
    print('m_ret', m_ret)
    return m_ret

while(True):
    count+=1

    ok, frame = cap.read()
    if not ok:
        break

    # Identify mode
    if identify and len(L)>0:
        M =[]
        # Compute keypoints and descriptors of current frame
        keypoints, descriptors = orb.detectAndCompute(frame[:,:,1],None)
        for i in range(len(L)):
            M.append(get_ransac_matches(keypoints,descriptors,K[i], D[i]))

        cv2.putText(frame, N[np.argmax(M)], (0,int(im_rows*.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.putText(frame,'matches:', (0,int(im_rows*.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        Ms = [str(m) for m in M]
        for i in range(len(Ms)):
            cv2.putText(frame, Ms[i], (150+i*50,int(im_rows*.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    cv2.imshow('frame',frame)

    k = cv2.waitKey(1)

    if k == ord('i'):
        identify = not identify

    if k == ord('q'):
        break

    # Save prototypes
    if k == ord('w'):
        L.append(frame)
        N.append('object ' +str(len(L)))
        cv2.imshow('frame'+str(len(L)),frame)
        keypoints, descriptors = orb.detectAndCompute(frame[:,:,1],None)
        K.append(keypoints)
        D.append(descriptors)
        print('Captured',len(L),'object prototypes')
    
    # Get prototypes from files
    if k == ord('f'):
        for img_name in imgs_to_read:
            img = cv2.imread(img_name)
            L.append(img)  
            N.append('object ' +str(len(L)))
            cv2.imshow('frame'+str(len(L)),frame)
            keypoints, descriptors = orb.detectAndCompute(img[:,:,1],None)
            K.append(keypoints)
            D.append(descriptors)
            print('Captured',len(L),'object prototypes')


elapsed_time = time.time()-start
print('Capture speed: {:.2f} frames per second'.format(count/elapsed_time))
cap.release()
cv2.destroyAllWindows()


