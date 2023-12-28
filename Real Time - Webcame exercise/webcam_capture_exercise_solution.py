import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage import transform as tr
from scipy import ndimage as nd

im_rows, im_cols = 360, 640
im_rows2, im_cols2 = 720, 960
cap = cv2.VideoCapture(0)
# We can set the size of the frames captured (default is 480x640)
cap.set(3,im_cols)
cap.set(4,im_rows)

start = time.time()
count, frame_count = 0,0
state = 0
n = 100
H = None
last_saved = -1
state_dict = {ord(str(k)):k for k in range(9)}
old_frame = 0
video_count = 0
thr = .03
second_cam_active = False

while(True):
    k = cv2.waitKey(1)

    if k in state_dict:
        state = state_dict[k]
        print('state changed to',state)

    count+=1

    ok, frame = cap.read()
    if not ok:
        break

    if state==1:
        frame = frame[:,::-1]
    elif state==2:
        frame[:,:,:-1] = 0
    elif state==3:
        curr = int(time.time())
        if curr!=last_saved:
            last_saved = curr
            k = ord('w')
    elif state==4:
        for ch in range(3):
            frame[:,:,ch]  = nd.gaussian_filter(frame[:,:,ch],sigma=5)
    elif state==5:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_name = 'video_out'+str(video_count)+'.avi'
        video_count+=1
        out = cv2.VideoWriter(video_name,fourcc, 20.0, (im_cols,im_rows))
        for i in range(n):
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1)
            out.write(frame)
            ok, frame = cap.read()
        out.release()
        print('wrote video to',video_name)
        state = 0
    elif state==6:
        frame = np.float32(frame)/255.
        if H==None:
            img = np.float32(plt.imread('tv_f22.jpg'))/255.
            tv_points = np.array([[749,145],[934,88],[933,301],[750,293]])
            source_points = np.array([[0,0],[frame.shape[1]-1,0],[frame.shape[1]-1,frame.shape[0]-1],[0,frame.shape[0]-1]])
            H = tr.ProjectiveTransform()
            H.estimate(source_points, tv_points)
            mask = tr.warp(frame*0+1, H.inverse, output_shape=img.shape)
            masked_im = img[:,:,::-1]*(1-mask)
        frame_warped = tr.warp(frame, H.inverse, output_shape=img.shape)
        frame = masked_im + frame_warped*mask
    elif state==7:
        frame = np.float32(frame)/255.
        diff = np.sqrt(np.mean((frame-old_frame)**2))
        if diff>thr:
            k = ord('w')
        old_frame = frame
    elif state==8:
        if not second_cam_active:
            print('Initializing second camera')
            cap2 = cv2.VideoCapture(1)
            #cap2.set(3,im_cols)
            #cap2.set(4,im_rows)
            second_cam_active = True
        ok, frame2 = cap2.read()
        if not ok:
            break

        cv2.imshow('frame 2',frame2)

    if k == ord('q'):
        break

    if k == ord('w'):
        frame_count+=1
        plt.imsave('img'+str(1000+frame_count)[1:]+'.jpg',frame[:,:,::-1])
        print('saved frame',frame_count)

    cv2.imshow('frame',frame)

elapsed_time = time.time()-start
print('Capture speed: {:.2f} frames per second'.format(count/elapsed_time))
cap.release()
cv2.destroyAllWindows()
