import gc
import numpy as np

class StegoModel:
    """
    A versatile and framework-agnostic model loader for performing steganography operations.

    This class enables loading pre-trained models from either TensorFlow or PyTorch frameworks,
    supporting both hiding (embedding) a secret image into cover images and revealing (extracting)
    secret images from stego images. It supports batch processing for efficient computation.

    Attributes:
        framework (str): The deep learning framework used ('tensorflow' or 'pytorch').
        device (str): Computation device, e.g., 'cpu' or 'cuda'.
        model: Loaded deep learning model.
        custom_objects: Custom objects to load the model (only for TensorFlow).
        post_process_hide_func (callable, optional): Optional post-processing function to apply
            to the raw output of the hide method.

    Args:
        model_path (str): Path to the pre-trained model file.
        framework (str): Framework of the model ('tensorflow' or 'pytorch'). Default is 'tensorflow'.
        device (str): Device to load and execute the model. Default is 'cpu'.
        custom_objects: Custom objects to load the model (only for TensorFlow).
    """
    def __init__(self, 
                model_path:str,
                framework:str, 
                device:str='cpu',
                custom_objects=None,
                post_process_hide_func=None):
        
        self.model_path = model_path
        self.device = device
        self.framework = framework
        self.custom_objects = custom_objects
        self.model = self.load_model()
        self.post_process_hide_func = post_process_hide_func
    
    def load_model(self):
        # load the model
        if self.framework == 'tensorflow':
            import tensorflow as tf
            model = tf.keras.models.load_model(self.model_path, 
                                               custom_objects=self.custom_objects)
            
            return model
        elif self.framework == 'pytorch':
            import torch
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()

            return model
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
             
    def hide(self, cover_images, secret, batch_size=8):
        """
        Hides a secret image into a cover image using the pre-trained model.

        Args:
            cover_images (list of np.ndarray): Cover images to hide the secret image into.
            secret (np.ndarray): Secret image to hide into the cover images.
            batch_size (int): Batch size for processing multiple images simultaneously.
            kwargs: Additional keyword arguments to pass to the post-processing function.

        Returns:
            np.ndarray: Stego images containing the secret images.
        """
        num_images = len(cover_images)
        cover_images = np.array(cover_images)
        secrets = np.tile(secret, (num_images, 1, 1, 1))

        batch_padding = (batch_size - (num_images % batch_size)) % batch_size
        if batch_padding > 0:
            filler = np.zeros((batch_padding,) + cover_images.shape[1:], dtype=cover_images.dtype)
            cover_images = np.concatenate((cover_images, filler), axis=0)
            secrets= np.concatenate((secrets, filler), axis=0)

        if self.framework == 'tensorflow':
            stegos = self.model.predict([secrets, cover_images], 
                                        batch_size=batch_size, verbose=True)
        elif self.framework == 'pytorch':
            with torch.no_grad():
                covers_torch = torch.from_numpy(cover_images).to(self.device)
                secretos_torch = torch.from_numpy(secrets).to(self.device)
                stegos = self.model(input=secretos_torch, cover=covers_torch)
                stegos = stegos.cpu().numpy()

        # Remove any filler data.
        stegos = stegos[:num_images]

        # If a post-processing function was provided, apply it.
        if self.post_process_hide_func is not None:
            processed_stegos = self.post_process_hide_func(stegos)
            return processed_stegos
        else:
            return {"stego_images": stegos}
    
    def reveal(self, stego_images, batch_size=8):
        """
        Reveals a secret image from stego images using the pre-trained model."
        """
        num_images = len(stego_images)
        batch_padding = (batch_size - (num_images % batch_size)) % batch_size
        if batch_padding > 0:
            filler = np.zeros((batch_padding,) + stego_images.shape[1:], dtype=stego_images.dtype)
            stego_images = np.concatenate((stego_images, filler), axis=0)

        if self.framework == 'tensorflow':
            secrets = self.model.predict(stego_images, batch_size=batch_size, verbose=True)
        elif self.framework == 'pytorch':
            with torch.no_grad():
                stegos_torch = torch.from_numpy(stego_images).to(self.device)
                secrets = self.model(input=stegos_torch)
                secrets = secrets.cpu().numpy()

        return secrets[:num_images]
    
    def gpu_memory_reset(self):
        """
        Resets the GPU memory.
        """
        if self.framework == 'tensorflow':
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()
            tf.config.experimental.reset_memory_stats('GPU:0')

        elif self.framework == 'pytorch':
            import torch
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        print("GPU memory has been reset.")
        
    def gpu_memory_usage(self):
        """
        Returns the GPU memory usage.
        """
        print("Values are in GB.")
        if self.framework == 'tensorflow':
            import tensorflow as tf
            memory_usage = tf.config.experimental.get_memory_info('GPU:0')
            memory_usage['current'] = memory_usage['current'] / 1024**3
            memory_usage['peak'] = memory_usage['peak'] / 1024**3
            return memory_usage
        elif self.framework == 'pytorch':
            import torch
            return torch.cuda.memory_allocated()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
    