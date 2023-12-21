import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tr

plt.close('all')

img = mpimg.imread('utep_axe_02.jpg')
img = img/np.amax(img)

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.tight_layout()

source = [[0,0],[img.shape[1]-1,0], [img.shape[1]-1,img.shape[0]-1], [0,img.shape[0]-1]]
dest  = source.copy()

while True:
    print("Click source and destination of warp point, click near top-left corner to stop entering points")
    p = np.asarray(plt.ginput(n=2), dtype=np.int)
    if np.sum(p)< 100: # Less than 100. Top left corner
        break
    source.append(p[0])
    dest.append(p[1])
    plt.plot(p[:,0],p[:,1],marker = '*',color='y')
    plt.pause(0.01)
    plt.show()

source = np.array(source)
dest = np.array(dest)

H = tr.PiecewiseAffineTransform()
H.estimate(source, dest)
img_warped = tr.warp(img, H.inverse)

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
ax[0].imshow(img)
ax[1].imshow(img_warped)
ax[0].set_title('Original image')
ax[1].set_title('Transformed image')
plt.tight_layout()

# Plot control points
fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
ax[0].imshow(img)
ax[1].imshow(img_warped)
ax[0].set_title('Original image')
ax[1].set_title('Transformed image')
plt.tight_layout()
ax[0].plot(source[:,0],source[:,1],'*r')
ax[1].plot(dest[:,0],dest[:,1],'*r')

