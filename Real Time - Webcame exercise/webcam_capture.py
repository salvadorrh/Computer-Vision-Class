# Capture and display images from webcam
# Press 'q' to exit
# Press 'w' to write current image to jpg file

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from scipy import ndimage as nd
#from skimage import transform as tr

cap = cv2.VideoCapture(0)
# We can set the size of the frames captured (default is 480x640)
#cap.set(3,1280)
#cap.set(4,720)
cap.set(4,360)


start = time.time()
count, frame_count = 0,0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('delme.avi',fourcc, 20.0, (640,360))

state = 0
time_3 = time.time()
count_3 = 0
count_5 = 0
count_7 = 0
frame_base = 0
while(True):
    count+=1

    ok, frame = cap.read()
    if not ok:
        break

    k = cv2.waitKey(1)
    
    if state == 1:
        frame = frame[::,::-1,]
        cv2.imshow('frame',frame)
    elif state == 2:
        frame[::,::,0] = 0
        frame[::,::,1] = 0
        cv2.imshow('frame',frame)
    elif state == 3:
        curr_time = time.time()
        if curr_time - time_3 > 1.0:
            print(curr_time - time.time())
            plt.imsave('img'+str(1000+count_3)[1:]+'.jpg',frame[:,:,::-1])
            print('saved frame',count_3)
            count_3 += 1
            time_3 = time.time()
        cv2.imshow('frame',frame)
    elif state == 4:
        for ch in range(3):
            frame[:,:,ch] = nd.gaussian_filter(frame[:,:,ch],5)
        cv2.imshow('frame',frame)
    elif state == 5:
        n= 100
        while(count_5 < n):
            ok, frame = cap.read()
            if not ok:
                break
            count_5+=1
            out.write(frame)
        state = 0
        count_5 = 0
    elif state == 6:
        """
        tv_points = np.array([[749,145],[934,88],[933,301],[750,293]])
        img = plt.imread('tv.jpg')
        img = img/np.amax(img)
        img = img[:,:,::-1]
        source_points = np.array([[0,0],[img.shape[1]-1,0],[img.shape[1]-1,img.shape[0]-1],[0,img.shape[0]-1]])

        H1 = tr.ProjectiveTransform()

        H1.estimate(source_points, tv_points)
        img_warped = tr.warp(frame, H1.inverse,output_shape=(img.shape[0],img.shape[1]))
        mask = tr.warp(img*0+1,H1.inverse,output_shape=(img.shape[0],img.shape[1]))
        img_warped = mask*img_warped
        ones = np.ones_like(mask)

        mask2 = tr.warp(ones*0+1,H1.inverse,output_shape=(img.shape[0],img.shape[1]))
        mask3 = 1-mask2
        result = img*mask3 + img_warped
        cv2.imshow('frame',result)"""
    elif state == 7:
        diff = np.sum((frame_base - frame) ** 2)
        
        if diff > 32500000:
            print('diff', diff)
            frame_base = frame
            plt.imsave('img'+str(1000+count_7)[1:]+'.jpg',frame[:,:,::-1])
            print('saved frame: ',count_7)
            count_7 +=1
            
    else:
        cv2.imshow('frame',frame)

    if k == ord('q'):
        break

    if k == ord('w'):
        frame_count+=1
        plt.imsave('img'+str(1000+frame_count)[1:]+'.jpg',frame[:,:,::-1])
        #print(frame.shape)
        print('saved frame',frame_count)
    
    if k == ord('1'):
        state = 1
    if k == ord('2'):
        state = 2
    if k == ord('3'):
        time_3 = time.time()
        state = 3
    if k == ord('4'):
        state = 4
    if k == ord('5'):
        state = 5
    if k == ord('6'):
        state = 6
    if k == ord('7'):
        state = 7
        frame_base = frame

elapsed_time = time.time()-start
print('Capture speed: {:.2f} frames per second'.format(count/elapsed_time))
cap.release()
cv2.destroyAllWindows()
