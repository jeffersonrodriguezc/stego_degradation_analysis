import cv2
import numpy as np

def mask_from_points(size, points, radius=10):
    """ Create a mask of supplied size from supplied points
    :param size: tuple of output mask size
    :param points: array of [x, y] points
    :param radius: int of radius around points to include
    :returns: mask of values 0 and 255 where
              255 indicates the convex hull containing the points
    """
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.erode(mask, kernel)

    return mask


def overlay_image(foreground_image, mask, background_image):
    """ Overlay foreground image onto the background given a mask
    :param foreground_image: foreground image points
    :param mask: [0-255] values in mask
    :param background_image: background image points
    :returns: image with foreground where mask > 0 overlaid on background image
    """
    foreground_pixels = mask > 0
    background_image[..., :3][foreground_pixels] = foreground_image[..., :3][foreground_pixels]
    return background_image


def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img = np.copy(img)
    num_channels = 3
    for c in range(num_channels):
        masked_img[..., c] = img[..., c] * (mask / 255)

    return masked_img


def weighted_average(img1, img2, percent=0.5):
    if percent <= 0:
        return img2
    elif percent >= 1:
        return img1
    else:
        return cv2.addWeighted(img1, percent, img2, 1 - percent, 0)


def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):

    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1 - mask)

    return result_img