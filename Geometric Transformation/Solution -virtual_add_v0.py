import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
plt.rc('image', cmap='gray')

plt.close('all')

img = plt.imread('sport-tennis2.jpeg')
img = (img/np.amax(img)).astype(np.float32)

fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(img)
fig.suptitle('Original image', fontsize=14)

print("Click four source points")
dest = np.asarray(plt.ginput(n=4))

img2 = plt.imread('vll_utep_720.jpg')
img2 = (img2/np.amax(img2)).astype(np.float32)


fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(img2)

src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])

H1 = tf.ProjectiveTransform()
H1.estimate(src, dest)
warped = tf.warp(img2, H1.inverse, output_shape=(img.shape[0],img.shape[1]))
warp_mask = tf.warp(img2*0+1, H1.inverse, output_shape=(img.shape[0],img.shape[1]))


fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].imshow(img2)
ax[1].imshow(warped)
ax[0].set_title('Source image')
ax[1].set_title('Transformed source image')

court_mask = img - np.array([.19,.38,.59])
court_mask = np.sum(court_mask**2,axis=2)
court_mask = court_mask - np.amin(court_mask)
court_mask = court_mask/np.amax(court_mask)<0.05
print(court_mask.shape)
combined_mask = warp_mask * np.expand_dims(court_mask,axis=2)

fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
ax[0].imshow(warp_mask)
ax[1].imshow(court_mask)
ax[2].imshow(combined_mask)
ax[0].set_title('Warp mask')
ax[1].set_title('Court mask')
ax[2].set_title('Combined mask')

print('Shape combined_mask', combined_mask.shape)
print('Shape img', img.shape)
combined = warped*combined_mask + img*(1-combined_mask)

plt.figure()
plt.title("warped *")
plt.imshow(warped*combined_mask)

plt.figure()
plt.title("warped")
plt.imshow(warped)

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
ax[0].imshow(img)
ax[1].imshow(combined)
ax[0].set_title('Original')
ax[1].set_title('Final')
plt.tight_layout()
