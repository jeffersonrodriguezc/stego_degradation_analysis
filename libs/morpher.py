import cv2

import numpy as np
import pandas as pd

from pathlib import Path

from libs.locator import weighted_average_points, face_points
from libs.warper import warp_image
from libs.blender import weighted_average, mask_from_points
from libs.aligner import resize_align


def load_image_points(image: str | np.ndarray, size: tuple[int, int]) -> tuple:
    """
    Load the image and detect the face points

    :param image: The image to load
    :type image: str | np.ndarray
    :param size: The size to resize the image to
    :type size: tuple[int, int]

    :return: The image and the face points
    :rtype: tuple
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    points = face_points(img)
    
    if len(points) != 0:
        return resize_align(img, points, size)
    else:
        raise ValueError("No face detected in the image")


def morph(
    src_path: str | Path,
    dst_path: str | Path,
    filename: str | None = None,
    alpha: float = 0.5,
    background: str = "black",
    size: tuple[int, int] | None = None,
    src_points=None,
    dst_points=None,
    return_morph=False,
):
    # Loading the images
    src_img = cv2.imread(str(src_path))
    dst_img = cv2.imread(str(dst_path))

    if not size:
        size = (
            min(src_img.shape[0], dst_img.shape[0]),
            min(src_img.shape[1], dst_img.shape[1]),
        )

    # Loading the points
    aux_src = src_img.copy()
    aux_dst = dst_img.copy()

    if not src_points:
        src_img, src_points, src_scale, src_box = load_image_points(src_img, size)

    if not dst_points:
        dst_img, dst_points, dst_scale, dst_box = load_image_points(dst_img, size)

    
    # Warping the images
    if not isinstance(alpha,(int,float)):
        alpha = float(alpha[0])

    points = weighted_average_points(src_points, dst_points, alpha)  # Intermediate points
    src_face = warp_image(src_img, src_points, points, size)  # Warping the source image to the intermediate points
    end_face = warp_image(dst_img, dst_points, points, size)  # Warping the destination image to the intermediate points
    average_face = weighted_average(src_face, end_face, alpha)  # Averaging the warped images

    # Selecting the background type
    
    if background == "transparent":
        # Create a mask for the average face
        mask = mask_from_points(size=average_face.shape[:2], points=points)
        # Add the mask to the average face  
        average_face = np.dstack((average_face, mask)) 

    elif background == "average":
        # Create a mask for the average face
        mask = mask_from_points(size=average_face.shape[:2], points=points)
        # Create the average background
        average_background = weighted_average(src_img, dst_img, alpha)
        # Calculate the center of the points
        center = (int(np.mean(points[:, 0])), int(np.mean(points[:, 1])))
        # Clone the average face onto the source image
        average_face = cv2.seamlessClone(
            src=average_face, 
            dst=average_background, 
            mask=mask, 
            p=center,
            flags=cv2.NORMAL_CLONE
            )

    elif background == "seamless":
        # Inverse warp the average face
        iw_face = warp_image(average_face, points, src_points, size)
        # Create a mask for the inverse warped face
        mask = mask_from_points(iw_face.shape[:2], src_points, radius=30)

        # Taking as center the average of the points from the source image
        cy, cx = (
            round(np.mean(src_points[:, 0]), 0),
            round(np.mean(src_points[:, 1]), 0),
        )

        # Getting the center of the face
        center = int(cy), int(cx)

        # Clone the average face onto the source image
        average_face = cv2.seamlessClone(
            src=average_face, 
            dst=src_img, 
            mask=mask, 
            p=center,
            flags=cv2.NORMAL_CLONE
        )
    
    else:
        raise ValueError(
            "Background must be one of 'transparent', 'average', 'seamless'"
        )

    if not return_morph:
        print(filename)
        av = average_face.copy()
        av = cv2.resize(src=av, dsize=(int(av.shape[1]/src_scale), int(av.shape[0]/src_scale)))
        aux_src = aux_src.copy()
        aux_src[int(src_box[1]/src_scale):int(src_box[1]/src_scale)+av.shape[0], int(src_box[4]/src_scale):int(src_box[4]/src_scale)+av.shape[1]] = av
        cv2.imwrite(filename, aux_src)
    else:
        return average_face


def get_points(data: pd.DataFrame, fname):
    """
    Get the points from the data

    :param data: The data to get the points from
    :type data: pd.DataFrame
    :param fname: The filename
    :type fname: str

    :return: The points
    """
    x = data[data["fname"] == fname]["x"].values
    y = data[data["fname"] == fname]["y"].values

    return np.column_stack((x, y)).round().astype(np.int32)
