import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(4,360)

start = time.time()
count=0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('delme.avi',fourcc, 20.0, (640,360))

n= 300
while(count < n):

    ok, frame = cap.read()
    if not ok:
        break

    count+=1
    out.write(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

elapsed_time = time.time()-start
print('Capture speed: {:.2f} frames per second'.format(count/elapsed_time))
cap.release()
out.release()
cv2.destroyAllWindows()

