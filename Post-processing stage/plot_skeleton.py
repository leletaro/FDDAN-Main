

from skimage import io
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.morphology import skeletonize, thin
from skimage.filters import threshold_otsu

image = io.imread( "MaskRDD-140_Last\China_MotorBike_000176_masked_1.png", as_gray=True)

# 二值化图像
thresh = threshold_otsu(image)
binary = image > thresh

# # 应用 skeletonize 和 thin
# skeleton = skeletonize(image)
# thinned = thin(image)
# thinned_partial = thin(image, max_num_iter=5)

# 应用 skeletonize 和 thin
skeleton = skeletonize(binary)
thinned = thin(binary)
thinned_partial = thin(binary, max_num_iter=15)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].set_title('skeleton')
ax[1].axis('off')

ax[2].imshow(thinned, cmap=plt.cm.gray)
ax[2].set_title('thinned')
ax[2].axis('off')

ax[3].imshow(thinned_partial, cmap=plt.cm.gray)
ax[3].set_title('partially thinned')
ax[3].axis('off')

fig.tight_layout()
plt.show()

# skeleton,thinned and partially thinned