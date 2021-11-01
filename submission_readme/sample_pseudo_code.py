import os
import numpy as np

with open('testing_img_order.txt') as f:
     test_images = [x.strip() for x in f.readlines()]  # all the testing images

submission = []
for img in test_images:  # image order is important to your result
    predicted_class = your_model(img)  # the predicted category
    submission.append([img, predicted_class])

np.savetxt('answer.txt', submission, fmt='%s')