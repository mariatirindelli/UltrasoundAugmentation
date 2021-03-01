import us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Decay:

    def __init__(self, decay_function='constant', direction='upper'):

        assert decay_function in ['constant', 'linear', 'gaussian'], \
            "Decay function must be either 'constant' or 'linear' "
        assert direction in ['upper', 'lower', 'left', 'right'], "Decay function must be either 'upper' or 'lower' "

        self.decay_function = decay_function
        self.direction = direction

    @staticmethod
    def _linear_decay(initial_value, final_value, num_samples):
        return np.linspace(initial_value, final_value, num_samples)

    @staticmethod
    def _constant_decay(constant_value, num_samples):
        return np.ones(num_samples) * constant_value

    @staticmethod
    def _upper_linear_decay(line, array_to_fill, v1, v2):
        array_to_fill[0:line[1], line[0]] = Decay._linear_decay(v1, v2, line[1])

    @staticmethod
    def _lower_linear_decay(line, array_to_fill, v1, v2):
        array_to_fill[line[1]+1::, line[0]] = Decay._linear_decay(v1, v2, array_to_fill.shape[0] - (line[1] + 1))

    @staticmethod
    def _upper_constant_decay(line, array_to_fill, v1, v2=None):
        array_to_fill[0:line[1], line[0]] = us.Decay._constant_decay(v1, line[1])

    @staticmethod
    def _lower_constant_decay(line, array_to_fill, v1, v2=None):
        array_to_fill[line[1]+1::, line[0]] = us.Decay._constant_decay(v1, line[1])

    @staticmethod
    def _gaussian_left_tail(a, b, max_len, displacement_gradient, sigma=50):
        """
            :param a: Intensity at the Gaussian tail
            :param b: Intensity at the Gaussian peak
        .. image:: ../img/gaussian_left_tail.png
        """

        x = np.linspace(start=-500,
                        stop=0,
                        num=501,
                        endpoint=True)

        g = Deformation.gaussian(x, 0, sigma)
        norm_g = (b - a) * (g - np.min(g)) / (np.max(g) - np.min(g)) + a

        x_start_idxes = np.argwhere(np.abs(norm_g - a) < displacement_gradient).flatten()
        x_start_idx = 0 if x_start_idxes.size == 0 else max(x_start_idxes[0], 500 - max_len)

        gaussian_profile = norm_g[x_start_idx:-1]

        return gaussian_profile

    @staticmethod
    def _gaussian_right_tail(a, b, max_len, displacement_gradient, sigma=50):
        """
        :param a: Intensity at the Gaussian tail
        :param b: Intensity at the Gaussian peak
        .. image:: ../img/gaussian_right_tail.png
        """
        # lower coincides with right
        x = np.linspace( start=0,
                         stop=500,
                         num=501,
                         endpoint=True)

        g = Deformation.gaussian(x, 0, sigma)

        norm_g = (b - a) * (g - np.min(g)) / (np.max(g) - np.min(g)) + a

        x_start_idxes = np.argwhere(np.abs(norm_g - a) < displacement_gradient).flatten()

        x_start_idx = max_len if x_start_idxes.size == 0 else min(x_start_idxes[0], max_len)
        gaussian_profile = norm_g[1:x_start_idx]

        return gaussian_profile

    @staticmethod
    def _right_gaussian_decay(line, displacement_field, x_l, max_len, min_visible_transition, sigma=50):
        transition_profile = Decay._gaussian_right_tail(a=line[1],
                                                        b=line[2],
                                                        max_len=max_len,
                                                        displacement_gradient=min_visible_transition,
                                                        sigma=sigma)
        x_r = x_l + len(transition_profile)
        displacement_field[int(line[0]), x_l + 1:x_r + 1] = transition_profile

    @staticmethod
    def _left_gaussian_decay(line, displacement_field, x_r, max_len, min_visible_transition, sigma=50):
        transition_profile = Decay._gaussian_left_tail(a=line[1],
                                                       b=line[2],
                                                       max_len=max_len,
                                                       displacement_gradient=min_visible_transition,
                                                       sigma=sigma)

        x_l = x_r - len(transition_profile)
        displacement_field[int(line[0]), x_l:x_r] = transition_profile

    def get_decay_function(self):
        if self.decay_function == 'gaussian' and self.direction == 'right':
            return self._right_gaussian_decay
        if self.decay_function == 'gaussian' and self.direction == 'left':
            return self._left_gaussian_decay
        if self.decay_function == 'linear' and self.direction == 'upper':
            return self._upper_linear_decay
        elif self.decay_function == 'linear' and self.direction == 'lower':
            return self._lower_linear_decay
        elif self.decay_function == 'constant' and self.direction == 'upper':
            return self._upper_constant_decay
        elif self.decay_function == 'constant' and self.direction == 'lower':
            return self._lower_linear_decay
        else:
            raise ValueError("Unknown name or mode")

class Deformation(us.BaseMethod):
    def __init__(self,
                 lateral_transition_sigma=50,
                 min_visible_transition=0.01):

        super(Deformation, self).__init__()
        self.lateral_transition_sigma = lateral_transition_sigma
        self.min_visible_transition = min_visible_transition

    @staticmethod
    def _deform(image, deformation_field):

        mapx_base, mapy_base = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        map_x = mapx_base
        map_y = mapy_base + deformation_field

        deformed_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_CUBIC)
        return deformed_image

    @staticmethod
    def _get_upper_profile(line):
        return np.argwhere(line > 0)[0][0]

    @staticmethod
    def _get_lower_profile(line):
        return np.argwhere(line > 0)[-1][-1]

    def get_bone_profile(self, label, mode='upper'):

        [x_tl, _], [x_tr, _] = self._get_bone_edges(label)
        if mode == 'upper':
            y_profile = np.apply_along_axis(self._get_upper_profile, axis=0, arr=label[:, x_tl: x_tr + 1])
        elif mode == 'lower':
            y_profile = np.apply_along_axis(self._get_lower_profile, axis=0, arr=label[:, x_tl: x_tr + 1])

        return np.arange(x_tl, x_tr+1), y_profile

    def add_displacement_above_bone(self, label, displacement_map, initial_displacement, final_displacement,
                                    displacement_decay_function='linear'):

        xy_lower_profile = np.array(self.get_bone_profile(label, mode='upper'))
        decay = us.Decay(decay_function=displacement_decay_function,
                         direction='upper')

        np.apply_along_axis(decay.get_decay_function(),
                            axis=0,
                            arr=xy_lower_profile,
                            array_to_fill=displacement_map,
                            v1=initial_displacement,
                            v2=final_displacement)

        return displacement_map

    def add_displacement_below_bone(self, label, displacement_map, initial_displacement, final_displacement,
                                    displacement_decay_function='linear'):

        xy_lower_profile = np.array(self.get_bone_profile(label, mode='lower'))
        decay = us.Decay(decay_function=displacement_decay_function,
                         direction='lower')

        np.apply_along_axis(decay.get_decay_function(),
                            axis=0,
                            arr=xy_lower_profile,
                            array_to_fill=displacement_map,
                            v1=initial_displacement,
                            v2=final_displacement)

        return displacement_map

    def add_left_transition_displacement(self, displacement_field, x_r):

        if x_r <= 0:
            return displacement_field

        decay = Decay(decay_function='gaussian', direction='left')
        max_len_transition_area = x_r

        # the value ranges of the transition
        transition_ranges = np.array([np.arange(0, displacement_field.shape[0]),  # y idxes
                                      displacement_field[:, 0],  # gaussian tail
                                      displacement_field[:, x_r]  # gaussian center
                                      ])

        np.apply_along_axis(decay.get_decay_function(),
                            axis=0,
                            arr=transition_ranges,
                            displacement_field=displacement_field,
                            x_r=x_r,
                            max_len=max_len_transition_area,
                            min_visible_transition=self.min_visible_transition,
                            sigma=self.lateral_transition_sigma)

        return displacement_field

    def add_right_transition_displacement(self, displacement_field, x_l):

        if x_l >= displacement_field.shape[1] - 1:
            return displacement_field

        decay = Decay(decay_function='gaussian', direction='right')
        max_len_transition_area = displacement_field.shape[1] - x_l

        # the value ranges of the transition
        transition_ranges = np.array([np.arange(0, displacement_field.shape[0]),  # y idxes
                                      displacement_field[:, -1],  # gaussian tail
                                      displacement_field[:, x_l]   # gaussian center
                                      ])

        np.apply_along_axis(decay.get_decay_function(),
                            axis=0,
                            arr=transition_ranges,
                            displacement_field=displacement_field,
                            x_l=x_l,
                            max_len=max_len_transition_area,
                            min_visible_transition=self.min_visible_transition,
                            sigma=self.lateral_transition_sigma)

        return displacement_field

    def execute(self, image, label_full, displacement=50):

        label = self._get_biggest_component(label_full)
        [x_tl, _], [x_tr, _] = self._get_bone_edges(label)

        displacement_field = np.ones(label.shape) * displacement

        # 1. Setting the deformation field to 0 where there is bone
        displacement_field[label == 1] = 0

        # 2. Setting displacement below bone to 0
        displacement_field = self.add_displacement_below_bone(label=label,
                                                              displacement_map=displacement_field,
                                                              initial_displacement=0,
                                                              final_displacement=0,
                                                              displacement_decay_function='constant')

        # 3. Setting the displacement above bone to a linear displacement going from the max displacement at the top
        # of the skin to the value 0 in correspondence with bones
        displacement_field = self.add_displacement_above_bone(label=label,
                                                              displacement_map=displacement_field,
                                                              initial_displacement=displacement,
                                                              final_displacement=0,
                                                              displacement_decay_function='linear')

        displacement_field = displacement - displacement_field

        # 4. Smoothing transition at the bone right
        displacement_field = self.add_right_transition_displacement(displacement_field=displacement_field,
                                                                    x_l=x_tr)

        # 5. Smoothing transition at the bone left
        displacement_field = self.add_left_transition_displacement(displacement_field=displacement_field,
                                                                   x_r=x_tl)

        deformed_image = self._deform(image, displacement_field)
        deformed_label_full = self._deform(label_full, displacement_field)

        return deformed_image, deformed_label_full
