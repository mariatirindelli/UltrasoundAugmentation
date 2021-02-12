import unittest
import python.us_augmentation as us
import matplotlib.pyplot as plt


class TestDeformation(unittest.TestCase):

    def test_gaussian_transition(self):

        a = 0
        b = 5
        max_len = 20
        displacement_gradient = 0.01

        method = us.Deformation()

        x_start, gaussian_profile = method._gaussian_transition(a, b, 'right', max_len, displacement_gradient)

        plt.plot(gaussian_profile)
        plt.hlines(y=a, xmin=-5, xmax=25)
        plt.hlines(y=a+displacement_gradient, xmin=-5, xmax=25)
        plt.show()

