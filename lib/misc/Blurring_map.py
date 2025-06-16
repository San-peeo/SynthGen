from lib.lib_dep import *

def Blurring_map(data, blur_factor):

    """
    Usage
    ----------
    Apply a blurring effect to a 2D map, keeping the original size;
    It performs by resizing it to a smaller resolution (see blur_factor) and then resizing it back to the original resolution.

    Parameters
    ----------
    data         : numpy.ndarray
                   Input 2D array (original data).
    blur_factor  : int
                   Factor by which the resolution of the map is reduced before resizing it back.

    Output
    ----------
    data_blurred : numpy.ndarray
                   Blurred version of the input 2D map.
    """

    dim_orig = np.shape(data)
    dim_blur = (int(dim_orig[0]/blur_factor), int(2*dim_orig[0]/blur_factor))

    data_resized = skimage.transform.resize(data, dim_blur, order=3, mode='reflect', anti_aliasing=True)
    data_blurred = skimage.transform.resize(data_resized, dim_orig, order=3, mode='reflect', anti_aliasing=True)

    return data_blurred




##########################################################################################################################
##########################################################################################################################
