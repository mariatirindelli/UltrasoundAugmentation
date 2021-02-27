import us_augmentation as us
import numpy as np
from scipy.stats import nakagami

class NakagamiNoise(us.BaseMethod):
    def __init__(self):

        super(NakagamiNoise, self).__init__()

    @staticmethod
    def _get_nakagami_noise(image_shape, nakagami_shape_parameter):

        nakagami_noise = nakagami.rvs(nu=nakagami_shape_parameter,
                                      size=image_shape)

        return nakagami_noise

    def execute(self, image, label, nakagami_shape_parameter=0.5):

        nakagami_noise = self._get_nakagami_noise(image_shape=image.shape,
                                                  nakagami_shape_parameter=nakagami_shape_parameter)

        image = image.astype(np.float)

        image += nakagami_noise*10

        return image, label
