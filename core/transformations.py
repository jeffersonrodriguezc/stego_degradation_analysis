import io
import cv2
import numpy as np
from PIL import Image

class Transformations:
    """
    Class to apply transformations to stego images.

    This class supports applying geometric and non-geometric transformations
    (except deepfake and morphing, which will be handled separately).

    The class is designed to be stateless regarding the image. It can be used 
    with a single image or a dataset (list of images). Each transformation is applied 
    independently to the original image(s); that is, transformations do not compound.
    
    Supported transformations include:
      - crop
      - resize
      - translation
      - flip
      - rotation
      - warp
      - salt_pepper_noise
      - gaussian_noise
      - compression
      - sharpening
      - gaussian_blurring
      - brightness_contrast
      - gamma_correction
      - dithering
      - adjust_saturation
    """

    def __init__(self):
        """
        Initialize the Transformations class.
        """
        # Define all available transformation methods.
        self.transformations = {
            'gaussian_noise': self.apply_gaussian_noise,
            'crop': self.apply_crop,
            'resize': self.apply_resize,
            'translation': self.apply_translation,
            'flip': self.apply_flip,
            'salt_pepper_noise': self.apply_salt_pepper_noise,
            'gaussian_blur': self.apply_gaussian_blur,
            'compression': self.apply_compression,
            'warp': self.apply_simple_warp,
            'sharpen': self.apply_sharpen,
            'rotation': self.apply_rotation,
            'brightness_contrast': self.apply_brightness_contrast,
            'gamma_correction': self.apply_gamma_correction,
            'dithering': self.apply_dithering,
            'adjust_saturation': self.apply_adjust_saturation
        }

    def change_cv2PIL(self,image):
        """
        Convert an OpenCV image to a PIL image.
        :param image: OpenCV image.
        :return: PIL image.
        """
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    def change_PILcv2(self,image):
        """
        Convert a PIL image to an OpenCV image.
        :param image: PIL image.
        :return: OpenCV image.
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def apply_crop(self, image, **kwargs):
        """
        Crop an image by simulating a zoom (in or out) while keeping the central region.

        Expected kwargs:
        - zoom_factor: A float representing the zoom factor (default: 1.0).
                        For zoom in, zoom_factor >= 1.0 (e.g., 2.0 crops a central region half the size).
                        For zoom out, zoom_factor < 1.0 (e.g., 0.8 pads the image to simulate zooming out).
        
        The method crops the central region based on the zoom_factor:
        - If zoom_factor >= 1.0: crops a smaller region from the center and resizes it back to the original dimensions.
        - If zoom_factor < 1.0: pads the image with a black background, then crops the central region.
        
        :param image: Original image.
        :return: Cropped (zoomed) image.
        """
        # Convert cv2 image to PIL if necessary
        original_is_cv2 = isinstance(image, np.ndarray)
        
        if original_is_cv2:
            image = self.change_cv2PIL(image)
        
        zoom_factor = kwargs.get('zoom_factor', 1.0)
        width, height = image.size

        if zoom_factor >= 1.0:
            # Zoom in: crop a central region smaller than the original image.
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)
            left = (width - new_width) // 2
            upper = (height - new_height) // 2
            crop_box = (left, upper, left + new_width, upper + new_height)
            cropped = image.crop(crop_box)
            # Resize cropped image back to original dimensions.
            image = cropped.resize((width, height), Image.Resampling.LANCZOS)
        else:
            # Zoom out: create a larger canvas, paste the original image at its center,
            # and then crop the central region with the original image dimensions.
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)
            new_image = Image.new(image.mode, (new_width, new_height), "white")
            paste_left = (new_width - width) // 2
            paste_upper = (new_height - height) // 2
            new_image.paste(image, (paste_left, paste_upper))
            # Crop the central region with the original image size.
            left_crop = (new_width - width) // 2
            upper_crop = (new_height - height) // 2
            crop_box = (left_crop, upper_crop, left_crop + width, upper_crop + height)
            image = new_image.crop(crop_box)
    
        if original_is_cv2:
            image = self.change_PILcv2(image)

        return image

    def apply_resize(self, image, **kwargs):
        """
        Resize an image while maintaining the aspect ratio and then resize it to a fixed size.

        Expected kwargs:
            - scale: Scale factor for proportional resizing (default: 0.9)
            - target_size: Tuple (width, height) for the final resize (default: (224, 224)).

        :param image: Original image.
        :return: Transformed image.
        """
        # Convert cv2 image to PIL if necessary
        original_is_cv2 = isinstance(image, np.ndarray)
        if original_is_cv2:
            image = self.change_cv2PIL(image)

        scale = kwargs.get('scale', 0.9)
        target_size = kwargs.get('target_size', (224, 224))

        # Get the original dimensions.
        width, height = image.size

        # Calculate new dimensions based on the scale factor.
        new_width = int(width * scale)
        new_height = int(height * scale)

        # First, perform the proportional resize.
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Then, resize to the fixed target size.
        image = resized_img.resize(target_size, Image.Resampling.LANCZOS)

        if original_is_cv2:
            image = self.change_PILcv2(image)

        return image

    def apply_translation(self, image, **kwargs):
        """
        Operation for translating an image.

        Expected kwargs:
          - shift_x, shift_y: Shifts along the x and y axes (fraccion number).
            0.1 = If we want 10% of the width or height

        :param image: Original image.
        :return: Transformed image.
        """
        h, w = image.shape[:2]
        shift_x = kwargs.get('shift_x', 0.1) # 0.1 by default
        shift_y = kwargs.get('shift_y', 0.1) # 0.1 by default
        tx_pixels = int(w * shift_x)
        ty_pixels = int(h * shift_y)
        # Translation matrix
        M = np.float32([[1, 0, tx_pixels],
                        [0, 1, ty_pixels]])
        
        # Apply the translation
        image = cv2.warpAffine(image, M, (w, h))
        return image
    
    def apply_rotation(self, image, **kwargs):
        """
        Operation for rotating an image.

        Expected kwargs:
          - angle: Rotation angle in degrees.

        :param image: Original image.
        :return: Transformed image.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        angle = kwargs.get('angle', 5) # 5 by default
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Apply the rotation
        image = cv2.warpAffine(image, M, (w, h))

        return image

    def apply_flip(self, image, **kwargs):
        """
        Operation for flipping an image.

        Expected kwargs:
          - flipCode: Flip mode (1 = 'horizontal', 0 = 'vertical', -1 = 'both').

        :param image: Original image.
        :return: Transformed image.
        """
        flipCode = kwargs.get('flipCode', 1)
        image = cv2.flip(image, flipCode)
        return image
    
    def apply_simple_warp(self, image, **kwargs):
        """
        Apply a simple warp to enhance the central region of a facial image.
        
        This method assumes the face is roughly centered in the image and uses a fixed
        perspective transform to emphasize the center and reduce the borders.
        
        Expected kwargs:
        - warp_factor: A float controlling the intensity of the warp.
                        Default is 1.0 (no warp). Values > 1.0 move the source points inward,
                        effectively zooming in on the center.
        
        :param image: Original image.
        :return: Warped image.
        """
        warp_factor = kwargs.get("warp_factor", 1.0)
        height, width = image.shape[:2]

        # For warp_factor = 1.0, no warping will occur.
        # For warp_factor > 1.0, calculate an inward offset based on the factor.
        # The offset is proportional to the image dimensions.
        offset_x = int(width * (warp_factor - 1) / (2 * warp_factor))
        offset_y = int(height * (warp_factor - 1) / (2 * warp_factor))

        # Define source points: a rectangle inset from the borders by the computed offset.
        src = np.float32([
            [offset_x, offset_y],
            [width - offset_x, offset_y],
            [width - offset_x, height - offset_y],
            [offset_x, height - offset_y]
        ])

        # Destination points: the full image corners.
        dst = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        # Compute the perspective transformation matrix.
        M = cv2.getPerspectiveTransform(src, dst)
        # Apply the perspective warp.
        image = cv2.warpPerspective(image, M, (width, height))

        return image

    def apply_salt_pepper_noise(self, image, **kwargs):
        """
        Apply salt & pepper noise to an image.
        
        Expected kwargs:
        - salt_prob: Probability for salt noise (default: 0.01).
        - pepper_prob: Probability for pepper noise (default: 0.01).
        
        :param image: Original image.
        :return: Noisy image.
        """
        # Detect if input is an OpenCV image (numpy array)
        original_is_cv2 = isinstance(image, np.ndarray)
        if original_is_cv2:
            # Convert OpenCV (BGR) image to PIL (RGB)
            image = self.change_cv2PIL(image)
        
        # Retrieve parameters with defaults
        salt_prob = kwargs.get('salt_prob', 0.01)
        pepper_prob = kwargs.get('pepper_prob', 0.01)
        
        # Convert PIL image to numpy array (RGB)
        image_array = np.array(image)
        noisy_image = image_array.copy()
        
        # Calculate total number of pixels (assuming image_array.shape = (H, W, C))
        total_pixels = image_array.size // image_array.shape[2]
        num_salt = int(salt_prob * total_pixels)
        num_pepper = int(pepper_prob * total_pixels)
        
        # Generate random coordinates for salt (white) pixels
        salt_coords = [np.random.randint(0, dim, num_salt) for dim in image_array.shape[:2]]
        noisy_image[salt_coords[0], salt_coords[1], :] = 255  # set salt pixels to white
        
        # Generate random coordinates for pepper (black) pixels
        pepper_coords = [np.random.randint(0, dim, num_pepper) for dim in image_array.shape[:2]]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0    # set pepper pixels to black
        
        # Convert numpy array back to a PIL image
        noisy_image = Image.fromarray(noisy_image.astype('uint8'))
        
        # Convert back to OpenCV format if the original image was in that format
        if original_is_cv2:
            noisy_image = self.change_PILcv2(noisy_image)
        
        return noisy_image
    
    def apply_gaussian_noise(self, image, **kwargs):
        """
        Apply Gaussian noise to an image.
        
        Expected kwargs:
        - mean: Mean of the noise (default: 0).
        - sigma: Standard deviation of the noise (default: 25).
        
        :param image: Original image.
        :return: Noisy image.
        """
        # Determine if the input is an OpenCV image (numpy array)
        original_is_cv2 = isinstance(image, np.ndarray)
        if original_is_cv2:
            # Convert OpenCV BGR image to PIL RGB image
            image = self.change_cv2PIL(image)
        
        # Retrieve parameters with default values
        mean = kwargs.get('mean', 0)
        sigma = kwargs.get('sigma', 25)
        
        # Convert the PIL image to a numpy array (RGB)
        image_array = np.array(image).astype('float32')
        
        # Generate Gaussian noise and add it to the image
        gaussian_noise = np.random.normal(mean, sigma, image_array.shape).astype('float32')
        noisy_image_array = image_array + gaussian_noise
        
        # Ensure the pixel values are within the valid range [0, 255]
        noisy_image_array = np.clip(noisy_image_array, 0, 255).astype('uint8')
        
        # Convert the numpy array back to a PIL image
        noisy_image = Image.fromarray(noisy_image_array)
        
        # Convert back to OpenCV format if the original image was in that format
        if original_is_cv2:
            noisy_image = self.change_PILcv2(noisy_image)
        
        return noisy_image
    
    def apply_compression(self, image, **kwargs):
        """
        Apply compression to an image using in-memory encoding.
        
        Expected kwargs:
        - compression_type: Compression format, e.g. "JPEG", "WebP", "PNG" (default: "JPEG").
        - quality: Compression quality (default: 50).
        
        :param image: Original image.
        :return: Compressed image.
        """
        # Detect if the input is an OpenCV image (numpy array)
        original_is_cv2 = isinstance(image, np.ndarray)
        if original_is_cv2:
            # Convert OpenCV (BGR) image to PIL (RGB)
            image = self.change_cv2PIL(image)
        
        # Retrieve parameters with default values
        compression_type = kwargs.get("compression_type", "JPEG")
        quality = kwargs.get("quality", 50)
        
        try:
            # Ensure the image is in RGB mode
            image = image.convert("RGB")
            # Create an in-memory buffer
            buffer = io.BytesIO()
            # Save the image into the buffer with the specified compression
            image.save(buffer, format=compression_type, quality=quality)
            buffer.seek(0)
            # Load the compressed image from the buffer
            compressed_image = Image.open(buffer)
        except Exception as e:
            print(f"Error in compression: {e}")
            return image  # If an error occurs, return the original image
        
        # Convert back to OpenCV format if needed
        if original_is_cv2:
            compressed_image = self.change_PILcv2(compressed_image)
        
        return compressed_image

    def apply_sharpen(self, image, **kwargs):
        """
        Apply a sharpen filter to an image.
        
        Expected kwargs:
        - intensity: An integer controlling the level of sharpening enhancement (default: 0).
        
        The sharpening operation uses a basic kernel:
        [[ 0, -1,  0],
        [-1, 5 + intensity, -1],
        [ 0, -1,  0]]
        This kernel enhances the image's details by amplifying the center pixel and subtracting its neighbors,
        thus accentuating edges and textures.
        
        :param image: Original image.
        :return: Sharpened image.
        """
        # Retrieve the intensity parameter with a default value of 0
        intensity = kwargs.get('intensity', 0)
        
        # Create the sharpening kernel
        kernel = np.array([[0, -1, 0],
                        [-1, 5 + intensity, -1],
                        [0, -1, 0]], dtype=np.float32)
        
        # Apply the kernel using cv2.filter2D
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened

    def apply_gaussian_blur(self, image, **kwargs):
        """
        Apply Gaussian blur to an image.
        
        Expected kwargs:
        - ksize: Kernel size for the blur operation (default: 3). 
                Note: ksize must be an odd number.
        
        This operation applies a Gaussian blur filter to smooth the image, reducing high-frequency detail.
        This is different from adding Gaussian noise: 
        - Gaussian blur smooths and reduces details by averaging pixels with a Gaussian kernel.
        - Gaussian noise adds random variations (noise) following a normal distribution.
        
        :param image: Original image.
        :return: Blurred image.
        """
        # Retrieve the kernel size parameter with a default value of 3    
        ksize = kwargs.get("ksize", 3)
        # Ensure ksize is odd; if it's even, increment by 1.
        if ksize % 2 == 0:
            ksize += 1

        # Apply the Gaussian blur using OpenCV
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        return blurred

    def apply_brightness_contrast(self, image, **kwargs):
        """
        Adjust the brightness and contrast of an image.

        Expected kwargs:
        - alpha: Contrast control (default: 0.8). It scales the pixel values.
        - beta: Brightness control (default: 30). It adds an offset to the pixel values.

        The transformation applied is:
            new_img = alpha * img + beta
        using cv2.convertScaleAbs, which scales the image pixel values and ensures the result is in the 8-bit range.
        
        Effect on the image:
        - Increasing alpha (>1.0) enhances contrast.
        - Increasing beta (>0) makes the image brighter.
        - Conversely, decreasing these values will lower contrast or darken the image.

        :param image: Original image.
        :return: Adjusted image.
        """
        # Retrieve the alpha and beta parameters with default values
        alpha = kwargs.get("alpha", 0.8)
        beta = kwargs.get("beta", 30)

        # Apply brightness and contrast adjustment
        new_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return new_img

    def apply_gamma_correction(self, image, **kwargs):
        """
        Apply gamma correction to an image.
        
        Expected kwargs:
        - gamma: Gamma value for correction (default: 2.2).
        
        Gamma correction adjusts the brightness of the image using a non-linear transformation.
        It builds a lookup table mapping each pixel value (0-255) as:
            new_pixel = ((pixel / 255.0) ** (1/gamma)) * 255
        This means:
        - If gamma < 1, the image becomes brighter.
        - If gamma > 1, the image becomes darker.
        
        :param image: Original image.
        :return: Gamma-corrected image.
        """
        # Retrieve the gamma parameter with a default value of 2.2       
        gamma = kwargs.get("gamma", 2.2)
        invGamma = 1.0 / gamma
        # Build the lookup table for all pixel values [0, 255]
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        # Apply gamma correction using the lookup table
        corrected = cv2.LUT(image, table)
        
        # Return the result in OpenCV format
        return corrected

    def apply_dithering(self, image, **kwargs):
        """
        Apply dithering to an image by reducing its color palette using an adaptive palette.
        
        Expected kwargs:
        - num_colors: Number of colors for the reduced palette (default: 64).
        
        This operation converts the image to a palette-based (P mode) image using an adaptive palette,
        then converts it back to RGB. The effect is a reduction in color depth that produces a posterized
        or dithered look. Fine gradients and subtle details are lost, and the overall image appears as if
        it has been quantized to a limited set of colors.
        
        Recommended values for testing the robustness:
        - Use a lower number of colors (e.g., 16 or 32) to simulate severe color reduction.
        - Use 64 (default) or 128 for moderate degradation.
        
        :param image: Original image.
        :return: Dithered image.
        """
        # Detect if the input image is in OpenCV format (numpy array)
        original_is_cv2 = isinstance(image, np.ndarray)
        if original_is_cv2:
            # Convert OpenCV (BGR) image to PIL (RGB)
            image = self.change_cv2PIL(image)
        
        num_colors = kwargs.get("num_colors", 64)
        
        # Convert the image to a palette image using an adaptive palette with the specified number of colors
        dithered = image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        # Convert back to RGB to obtain a full-color image with reduced color depth
        dithered = dithered.convert("RGB")
        
        # Convert the resulting PIL image to a numpy array
        #dithered_np = np.array(dithered)
        
        # If the original image was in OpenCV format, convert RGB back to BGR
        if original_is_cv2:
            dithered = self.change_PILcv2(dithered)
        
        return dithered

    def apply_adjust_saturation(self, image, **kwargs):
        """
        Adjust the saturation of an image in the HSV color space.
        
        Expected kwargs:
        - saturation_factor: Factor to multiply the saturation channel (default: 1.0).
                            Values < 1.0 decrease saturation; values > 1.0 increase saturation.
        
        The operation converts the image from BGR to HSV, adjusts the S channel by the given factor,
        and converts it back to BGR. This simulates variations in color intensity that can occur under
        different illumination conditions.
        
        Recommended values for testing:
        - saturation_factor: between 0.8 and 1.2.
        
        :param image: Original image.
        :return: Image with adjusted saturation.
        """
        # Convert from BGR to HSV.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Retrieve saturation factor.
        saturation_factor = kwargs.get("saturation_factor", 1.0)
        
        # Multiply the S channel by the factor and clip to valid range [0, 255].
        hsv[:,:,1] = np.clip(hsv[:,:,1].astype('float32') * saturation_factor, 0, 255).astype('uint8')
        
        # Convert back from HSV to BGR.
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return adjusted

    def apply_transformation(self, image, transformation_name, **kwargs):
        """
        Apply a single transformation to the provided image or dataset independently.

        :param image: A single image or a list of images.
        :param transformation_name: Name of the transformation to apply.
        :param kwargs: Parameters for the transformation.
        :return: Transformed image if a single image is provided, or a list of transformed images if a dataset.
        """
        if transformation_name not in self.transformations:
            raise ValueError(f"Transformation not found: {transformation_name}")

        def process(img):
            return self.transformations[transformation_name](img, **kwargs)

        if isinstance(image, list):
            return [process(img) for img in image]
        else:
            return process(image)

    def apply_pipeline(self, image, pipeline):
        """
        Apply a sequence (pipeline) of transformations independently to the provided image or dataset.
        
        Each transformation in the pipeline is applied separately to the original image(s). 
        That is, transformations are not compounded on top of each other.

        :param image: A single image or a list of images.
        :param pipeline: List of dictionaries, each with the format:
                         {'name': <transformation_name>, 'params': {<parameters>}}
                         Example:
                         [
                             {'name': 'crop', 'params': {'x': 0, 'y': 0, 'width': 100, 'height': 100}},
                             {'name': 'resize', 'params': {'width': 50, 'height': 50}},
                         ]
        :return: Dictionary mapping each transformation name to its results (transformed image or list of images).
        """
        results = {}
        for step in pipeline:
            transformation_name = step.get('name')
            params = step.get('params', {})
            results[transformation_name] = self.apply_transformation(image, transformation_name, **params)
        return results
