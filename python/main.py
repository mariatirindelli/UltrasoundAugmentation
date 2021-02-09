import cv2
import matplotlib.pyplot as plt
import us_augmentation

image = cv2.imread("..\\matlab\\test_data\\test_P4-2V\\real_img.png", 0)

image = us_augmentation.dummy_fun(image)

plt.imshow(image)
plt.show()

print()