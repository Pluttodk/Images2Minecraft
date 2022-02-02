import cv2
import numpy as np
import os
import urllib.request
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

scale = (15,15)
ideal_blocks = np.zeros(scale)

house = cv2.imread("target/house.jpg")
house = cv2.resize(house, (48*scale[0],48*scale[1]))
original_scale = house.shape

blocks = {}
for file in os.listdir("img/"):
    img = cv2.imread(f"img/{file}")
    if img.shape == (48,48,3):
        blocks[file] = img

# for i, line in enumerate(open("images.txt").readlines()):
#     urllib.request.urlretrieve(line, f"{i}.png")

blocks_shape = (48,48)

image_scaled = (scale[0]*blocks_shape[1],scale[1]*blocks_shape[1])

scores = {}

shrinked = cv2.resize(house, image_scaled)

total_white = np.ones((48,48,3))*255
blocks_value = list(blocks.values())
blocks_keys = list(blocks)
for xi, x in enumerate(range(0, image_scaled[0], blocks_shape[0])):
    for yi, y in enumerate(range(0, image_scaled[1], blocks_shape[1])):
        
        focus_area = shrinked[x:x+blocks_shape[0],y:y+blocks_shape[1]]
        score = [rmse(block, focus_area) for block in blocks.values()]
        best_bloc = np.argmin(score)
        shrinked[x:x+blocks_shape[0],y:y+blocks_shape[1]] = blocks_value[best_bloc]
#         if np.sum(focus_area) == np.sum(total_white):
#             continue

#         score = [np.sum(focus_area - block) for block in blocks_value]

                
cv2.imshow("", cv2.resize(shrinked, (original_scale[1]//2,original_scale[0]//2)))

cv2.waitKey(0)

cv2.destroyAllWindows()


    