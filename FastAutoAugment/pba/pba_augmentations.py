import random
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import numpy as np
import collections
random_mirror = True


PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


def create_cutout_mask(img_height, img_width, num_channels, size):
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0,
                                                       width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                   min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((num_channels, img_height, img_width), np.float32)
    mask[:, upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1]] = 0.
    return mask, upper_coord, lower_coord


def _enhancer_impl(enhancer):

    def impl(pil_img, level):
        v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


def int_parameter(level, maxval):
    return int(level * maxval / PARAMETER_MAX)


def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def auto_contrast(img, _):
    return ImageOps.autocontrast(img)


def invert(img, _):
    return ImageOps.invert(img)


def equalize(img, _):
    return ImageOps.equalize(img)


def color(img, level):
    return _enhancer_impl(ImageEnhance.Color)(img, level)


def contrast(img, level):
    return _enhancer_impl(ImageEnhance.Contrast)(img, level)


def brightness(img, level):
    return _enhancer_impl(ImageEnhance.Brightness)(img, level)


def sharpness(img, level):
    return _enhancer_impl(ImageEnhance.Sharpness)(img, level)


def rotate(img, level):
    degrees = int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees)


def posterize(img, level):
    level = int_parameter(level, 4)
    return ImageOps.posterize(img, 4 - level)


def shear_x(img, level):
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def shear_y(img, level):
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def translate_x(img, level,):
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def translate_y(img, level):
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def solarize(pil_img, level):
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img, 256 - level)


def cutout(img, level):
    size = int_parameter(level, 20)
    if size <= 0:
        return img
    img_height, img_width = img.size
    num_channels = 3
    _, upper_coord, lower_coord = (create_cutout_mask(img_height, img_width,
                                                      num_channels, size))
    pixels = img.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (125, 122, 113)  # set the colour accordingly
    return img


HP_TRANSFORMS = [
    rotate,
    translate_x,
    translate_y,
    brightness,
    color,
    invert,
    sharpness,
    posterize,
    shear_x,
    solarize,
    shear_y,
    equalize,
    auto_contrast,
    cutout,
    contrast
]

NAME_TO_TRANSFORM = collections.OrderedDict((t.__name__, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)


def get_augment_pba(name):
    return NAME_TO_TRANSFORM[name]


def apply_augment_pba(img, name, level):
    augment_fn = get_augment_pba(name)
    return augment_fn(img.copy(), level)