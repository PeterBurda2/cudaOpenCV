import cv2
import os


imgSRC = cv2.imread('data/frame-0.tif', cv2.IMREAD_UNCHANGED)

cx, cy = imgSRC.shape[1] // 2, imgSRC.shape[0] // 2

k = 0.4
nb_of_templates = 5

dx_max = int(k * cx)
dy_max = int(k * cy)

img_cut = imgSRC[cy - dy_max:cy + dy_max, cx - dx_max:cx + dx_max].copy()

cx_t = img_cut.shape[1] // 3
cy_t = img_cut.shape[0] // 2


img_template_1 = img_cut[:, 0 : cx_t].copy()
img_template_2 = img_cut[:, cx_t : (2 * cx_t)].copy()
img_template_3 = img_cut[:,  (2 * cx_t) :-1].copy()

img_template_4 = img_cut[0 : cy_t, :].copy()
img_template_5 = img_cut[cy_t : -1, :].copy()

for i in range(1, nb_of_templates + 1):
    cv2.imwrite(f"templates/template_{i}.tiff", eval(f"img_template_{i}"))
