import us_augmentation as us
import cv2
import numpy as np

class TiltingDeformation(us.BaseMethod):
    def __init__(self):

        super(TiltingDeformation, self).__init__()

    @staticmethod
    def _get_column_deformation(value, num_samples):

        deformation_profile = np.linspace(start=0,
                                          stop=value,
                                          num=num_samples)
        return deformation_profile

    def _get_deformation_field(self, image_shape, min_deformation, max_deformation):

        horizontal_top_deformation = np.linspace(max_deformation, min_deformation, image_shape[1])

        deformation_field = np.apply_along_axis(func1d=self._get_column_deformation,
                                                arr=horizontal_top_deformation,
                                                num_samples=image_shape[0], axis=0)

        return deformation_field

    @staticmethod
    def _deform(image, deformation_field):

        map_x_base, map_y_base = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        map_x = map_x_base
        map_y = map_y_base + deformation_field

        deformed_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_CUBIC)
        return deformed_image

    def execute(self, image, label, min_deformation=10, max_deformation=200):

        deformation_field = self._get_deformation_field(image_shape=image.shape,
                                                        min_deformation=min_deformation,
                                                        max_deformation=max_deformation)

        deformed_image = self._deform(image, deformation_field)
        deformed_label = self._deform(label, deformation_field)
        deformed_label = np.where(deformed_label > 0.5, 1, 0)

        return deformed_image, deformed_label
