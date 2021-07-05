import numpy as np
import random
import cv2 as cv
from genNoiseImage import genNoise

PATH_INPUT_IMAGE = r"/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/CovidAttendance/covid_attendance_20192.jpg"
REDUCTION_FACTOR = 0.7 
ITERATION_NUM = 5
PROB = 0.07


img = cv.imread(PATH_INPUT_IMAGE ,0) # Only for grayscale image
for i in range(ITERATION_NUM):
   new_rows = int(img.shape[0] * (REDUCTION_FACTOR)**i)
   new_cols = int(img.shape[1] * (REDUCTION_FACTOR)**i)
   new_img = cv.resize(img, (new_cols, new_rows), interpolation= cv.INTER_CUBIC)
   cv.imwrite(r"CovidAttendance/Origin/attendance_{}x{}".format(new_rows, new_cols) + ".jpg", new_img)
   new_img = genNoise(new_img, PROB)
   cv.imwrite(r"CovidAttendance/Noised/attendance_{}x{}".format(new_rows, new_cols) + ".jpg", new_img)

print ("===DONE!===")



