import os
import matplotlib.pyplot as plt
import cv2
import unittest
import us_augmentation as us
import random

test_filepath = os.path.abspath(__file__)
test_data_folder = os.path.join(os.path.split(test_filepath)[0], "test_data")

def load_image_label(root, img_name, label_name):
    image_path = os.path.join(root, img_name)
    label_path = os.path.join(root, label_name)

    image = cv2.imread(image_path, 0)
    label = cv2.imread(label_path, 0)
    return image, label

class TestTiltingDeformation(unittest.TestCase):
    root = os.path.join(test_data_folder, "us_augmentation")

    @staticmethod
    def plot(image, label, def_image, def_label):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        us.image_with_colorbar(fig, axs[0][0], image, cmap='gray', title="Image")
        us.image_with_colorbar(fig, axs[0][1], label, cmap=None, title="Label")

        us.image_with_colorbar(fig, axs[1][0], def_image, cmap='gray', title="Def-Image")
        us.image_with_colorbar(fig, axs[1][1], def_label, cmap=None, title="Def-Label")

        plt.tight_layout()
        plt.show()

    def test_execute(self):

        image, label = load_image_label(self.root, "linear_image.png", "linear_label.png")
        method = us.TiltingDeformation()

        for _ in range(10):
            min_deformation = random.randint(10, 200)
            max_deformation = random.randint(10, 200)

            def_image, def_label = method.execute(image=image,
                                                  label=label,
                                                  min_deformation=min_deformation,
                                                  max_deformation=max_deformation)

            self.plot(image, label, def_image, def_label)

    def test_execute_left_bone(self):
        image, label = load_image_label(self.root, "left_bone_image.png", "left_bone_label.png")

        method = us.TiltingDeformation()

        for _ in range(10):
            min_deformation = random.randint(10, 200)
            max_deformation = random.randint(10, 200)

            def_image, def_label = method.execute(image=image,
                                                  label=label,
                                                  min_deformation=min_deformation,
                                                  max_deformation=max_deformation)

            self.plot(image, label, def_image, def_label)

    def test_execute_left_right_bone(self):
        image, label = load_image_label(self.root, "right_bone_image.png", "right_bone_label.png")

        method = us.TiltingDeformation()

        for _ in range(10):
            min_deformation = random.randint(10, 200)
            max_deformation = random.randint(10, 200)

            def_image, def_label = method.execute(image=image,
                                                  label=label,
                                                  min_deformation=min_deformation,
                                                  max_deformation=max_deformation)

            self.plot(image, label, def_image, def_label)

    def test_execute_left_right_top_bone(self):
        image, label = load_image_label(self.root, "top_bone_image.png", "top_bone_label.png")

        method = us.TiltingDeformation()

        for _ in range(10):
            min_deformation = random.randint(10, 200)
            max_deformation = random.randint(10, 200)

            def_image, def_label = method.execute(image=image,
                                                  label=label,
                                                  min_deformation=min_deformation,
                                                  max_deformation=max_deformation)

            self.plot(image, label, def_image, def_label)

    def test_execute_left_right_multiple_bones(self):
        image, label = load_image_label(self.root, "multiple_bone_image.png", "multiple_bone_label.png")

        method = us.TiltingDeformation()

        for _ in range(10):
            min_deformation = random.randint(10, 200)
            max_deformation = random.randint(10, 200)

            def_image, def_label = method.execute(image=image,
                                                  label=label,
                                                  min_deformation=min_deformation,
                                                  max_deformation=max_deformation)

            self.plot(image, label, def_image, def_label)

