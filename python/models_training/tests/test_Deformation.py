import unittest
import cv2
import os
import us_augmentation as us
import matplotlib.pyplot as plt
import numpy as np

test_filepath = os.path.abspath(__file__)
test_data_folder = os.path.join(os.path.split(test_filepath)[0], "test_data")

def load_image_label(root, img_name, label_name):
    image_path = os.path.join(root, img_name)
    label_path = os.path.join(root, label_name)

    image = cv2.imread(image_path, 0)
    label = cv2.imread(label_path, 0)
    return image, label

class TestDeformation(unittest.TestCase):
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

    def test_gaussian_left_tail_1(self):

        a = 10
        b = 50
        max_len = 100
        displacement_gradient = 0.1
        lateral_transition_sigma = 50

        method = us.Decay()

        res = method._gaussian_left_tail(a, b, max_len, displacement_gradient, lateral_transition_sigma)
        plt.plot(res)
        plt.show()

    def test_gaussian_left_tail_2(self):

        a = 10
        b = -50
        max_len = 100
        displacement_gradient = 0.1
        lateral_transition_sigma = 50

        method = us.Decay()

        res = method._gaussian_left_tail(a, b, max_len, displacement_gradient, lateral_transition_sigma)
        plt.plot(res)
        plt.show()

    def test_gaussian_right_tail_1(self):

        a = 10
        b = 50
        max_len = 100
        displacement_gradient = 0.1
        lateral_transition_sigma = 50

        method = us.Decay()

        res = method._gaussian_right_tail(a, b, max_len, displacement_gradient, lateral_transition_sigma)
        plt.plot(res)
        plt.show()

    def test_gaussian_right_tail_2(self):

        a = 10
        b = -50
        max_len = 100
        displacement_gradient = 0.1
        lateral_transition_sigma = 50

        method = us.Decay()

        res = method._gaussian_right_tail(a, b, max_len, displacement_gradient, lateral_transition_sigma)
        plt.plot(res)
        plt.show()

    def test_add_displacement_above_bone(self):
        _, label = load_image_label(self.root, "linear_image.png", "linear_label.png")
        method = us.Deformation()
        displacement_map = np.zeros(label.shape)

        a = method.add_displacement_above_bone(label=label,
                                               displacement_map=displacement_map,
                                               initial_displacement=0,
                                               final_displacement=50,
                                               displacement_decay_function='linear')

        plt.imshow(a)
        plt.show()

    def test_add_displacement_below_bone(self):
        _, label = load_image_label(self.root, "linear_image.png", "linear_label.png")
        method = us.Deformation()
        displacement_map = np.zeros(label.shape)

        a = method.add_displacement_below_bone(label=label,
                                               displacement_map=displacement_map,
                                               initial_displacement=0,
                                               final_displacement=50,
                                               displacement_decay_function='linear')

        plt.imshow(a)
        plt.show()

    def test_add_right_transition_displacement(self):

        method = us.Deformation()
        displacement_field = np.load(os.path.join(test_data_folder,
                                                  "us_augmentation",
                                                  "not_smoothed_bone_displacement.npy"))
        displacement = 50
        x_tr = 131

        displacement_field = displacement - displacement_field

        displacement_field = method.add_right_transition_displacement(displacement_field=displacement_field,
                                                                      x_l=x_tr)

        plt.imshow(displacement_field)
        plt.show()

    def test_add_left_transition_displacement(self):

        method = us.Deformation()
        displacement_field = np.load(os.path.join(test_data_folder,
                                                  "us_augmentation",
                                                  "not_smoothed_bone_displacement.npy"))
        displacement = 50
        x_tl = 41

        displacement_field = displacement - displacement_field

        displacement_field = method.add_left_transition_displacement(displacement_field=displacement_field,
                                                                     x_r=x_tl)

        plt.imshow(displacement_field)
        plt.show()

    def test_execute_left_bone(self):
        image, label = load_image_label(self.root, "left_bone_image.png", "left_bone_label.png")
        method = us.Deformation()

        for i in range(10):
            displacement = int(np.random.uniform(30, 100))
            def_image, def_label = method.execute(image, label, displacement)

        # only plot last image
        self.plot(image, label, def_image, def_label)

    def test_execute_right_bone(self):
        image, label = load_image_label(self.root, "right_bone_image.png", "right_bone_label.png")
        method = us.Deformation()

        for i in range(10):
            displacement = int(np.random.uniform(30, 100))
            def_image, def_label = method.execute(image, label, displacement)

        self.plot(image, label, def_image, def_label)

    def test_execute_top_bone(self):
        image, label = load_image_label(self.root, "top_bone_image.png", "top_bone_label.png")
        method = us.Deformation()

        def_image, def_label = method.execute(image, label)
        for i in range(10):
            displacement = int(np.random.uniform(30, 100))
            def_image, def_label = method.execute(image, label, displacement)

        self.plot(image, label, def_image, def_label)

    def test_execute_bottom_bone(self):
        image, label = load_image_label(self.root, "bottom_bone_image.png", "bottom_bone_label.png")
        method = us.Deformation()

        for i in range(10):
            displacement = int(np.random.uniform(30, 100))
            def_image, def_label = method.execute(image, label, displacement)

        self.plot(image, label, def_image, def_label)

    def test_execute_multiple_bones(self):
        image, label = load_image_label(self.root, "multiple_bone_image.png", "multiple_bone_label.png")
        method = us.Deformation()

        for i in range(10):
            displacement = int(np.random.uniform(30, 100))
            def_image, def_label = method.execute(image, label, displacement)

        self.plot(image, label, def_image, def_label)




