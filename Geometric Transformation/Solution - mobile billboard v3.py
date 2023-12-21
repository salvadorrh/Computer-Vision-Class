import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tr

plt.close('all')

img = plt.imread('mobile billboard.jpg')
img = img/np.amax(img)

img2 = plt.imread('utep.jpg')
img2 = img2/np.amax(img2)

fig, ax = plt.subplots(figsize=(8,4))
ax.imshow(img2)
fig.suptitle('Image to insert', fontsize=14)

fig, ax = plt.subplots(figsize=(14,8))
ax.imshow(img)
plt.tight_layout()
fig.suptitle('Target image', fontsize=14)

print("Click four destination points of polygon")
dest = np.asarray(plt.ginput(n=4))

src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])

H1 = tr.ProjectiveTransform()
H1.estimate(src, dest)

warped = tr.warp(img2, H1.inverse, output_shape=(img.shape[0],img.shape[1]))
# It's like warping a white image
mask = tr.warp(img2*0+1, H1.inverse, output_shape=(img.shape[0],img.shape[1]))

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(warped)
ax[0].set_title('Warped image')
ax[1].imshow(mask)
ax[1].set_title('Mask')
plt.tight_layout()

combined = warped + img*(1-mask)
fig, ax = plt.subplots()
ax.imshow(combined)
fig.suptitle('Final image')
plt.tight_layout()

fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
ax[0].imshow(img)
ax[1].imshow(combined)
ax[0].set_title('Original')
ax[1].set_title('Final')
plt.tight_layout()