import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio  
from skimage.metrics import structural_similarity as ssim

class Metrics:
    """
    Class to calculate error metrics between two images.
    
    Attributes:
        image_a (numpy.ndarray): First RGB image.
        image_b (numpy.ndarray): Second RGB image.
    """
    def __init__(self, image_a, image_b):
        self.image_a = image_a
        self.image_b = image_b

    def compute_all(self):
        """
        Computes all error metrics: SSIM, MSE, and PSNR.
        
        Returns:
            tuple: (ssim_value, mse_value, psnr_value)
        """
        return {'ssim': self.compute_ssim(), 
                'mse': self.compute_mse(), 
                'psnr': self.compute_psnr()}

    def compute_mse(self):
        """
        Calculates the Mean Squared Error (MSE) between the corresponding pixels 
        of two RGB images.
        
        Returns:
            float: The MSE value.
        
        Raises:
            ValueError: If the images do not have the same dimensions.
        """
        if self.image_a.shape != self.image_b.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Compute the squared difference for all pixels and channels.
        err = np.sum((self.image_a.astype("float") - self.image_b.astype("float")) ** 2)
        err /= float(self.image_a.shape[0] * self.image_a.shape[1] * self.image_a.shape[2])  # Divide by total number of pixels and channels
        return err

    def compute_psnr(self):
        """
        Measures the Peak Signal-to-Noise Ratio (PSNR) between the two images.
        PSNR is computed using the maximum pixel value and the MSE.
        
        Returns:
            float: The PSNR value.
        """
        mse_value = self.compute_mse()
        if mse_value == 0:
            return float('inf')  # Infinite PSNR if there are no differences
        max_pixel = 255.0  # Maximum possible pixel value for an 8-bit image
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
        return psnr_value

    def compute_ssim(self):
        """
        Computes the Structural Similarity Index (SSIM) between two images.
        
        Returns:
            float: The SSIM value.
        
        Raises:
            ValueError: If the images do not have the same dimensions.
        """
        if self.image_a.shape != self.image_b.shape:
            raise ValueError("Images must have the same dimensions for SSIM calculation.")

        # Determine window size: use the minimum between 7 and the smallest dimension (if possible)
        min_dim = min(self.image_a.shape[0], self.image_a.shape[1])
        win_size = min(7, min_dim) if min_dim >= 7 else min_dim

        ssim_value = ssim(self.image_a, self.image_b, win_size=win_size, channel_axis=-1)
        return ssim_value
