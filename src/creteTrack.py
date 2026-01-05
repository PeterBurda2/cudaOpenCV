import cv2
import numpy as np
import matplotlib.pyplot as plt



imgSRC = cv2.imread('data/frame-0.tif', cv2.IMREAD_UNCHANGED)

cx, cy = imgSRC.shape[1] // 2, imgSRC.shape[0] // 2

k = 0.4
K = 0.9


dx, dy = cx, cy
t_dx, t_dy = int(k * dx), int(k * dy)
I_dx, I_dy = int(K * dx), int(K * dy)


R = 50 #px
theta = np.linspace(0, 2 * np.pi, 420)

cx_R = R * np.cos(theta) + cx
cy_R = R * np.sin(theta) + cy

cv2.imwrite(f'templates/t_1.tif', imgSRC[cy - t_dy:cy + t_dy, cx - t_dx:cx + t_dx])

for i in range(len(theta)):

    cv2.imwrite(f'series_1/frame_{i}.tif', imgSRC[int(cy_R[i]) - I_dy:int(cy_R[i]) + I_dy, int(cx_R[i]) - I_dx:int(cx_R[i]) + I_dx])


# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(imgSRC[cy - t_dy:cy + t_dy, cx - t_dx:cx + t_dx], cmap='gray')
# axs[1].imshow(imgSRC[cy - I_dy:cy + I_dy, cx - I_dx:cx + I_dx], cmap='gray')
# # axs[1].imshow(imgSRC, cmap='gray')


# plt.title('Original Image')
# plt.show() 






