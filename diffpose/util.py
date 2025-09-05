import numpy as np
import scipy

#对CT数据resample
def resample(image, spacing, new_spacing=[1,1,1]):
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing