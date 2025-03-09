import gc
import numpy as np
from typing import Optional, Callable, Dict

class StegoModel:
    """
    StegoModel is a class that enables hiding and revealing secret images 
    using pre-trained deep learning models. It supports different frameworks 
    and allows for custom preprocessing and postprocessing functions.

    Attributes:
        model_path_hide (str): Path to the pre-trained model used for hiding an image.
        model_path_reveal (str): Path to the pre-trained model used for revealing a hidden image.
        framework (str): The deep learning framework used (e.g., TensorFlow, PyTorch).
        device (str): The device where the model will run ('cpu' or 'cuda'). Default is 'cpu'.
        custom_objects (Optional[Dict]): Dictionary of custom objects required by the model (if any).
        objective (str): Defines the initial objective of the model. 
                         Options: 'hide' (default), 'reveal', or 'both'.
        post_process_hide_func (Optional[Callable]): A function to apply post-processing 
                                                     to the output of the hiding model.
        pre_process_reveal_func (Optional[Callable]): A function to apply pre-processing 
                                                      before passing the image to the reveal model.
        model (object): The currently loaded model (either hide or reveal).
        model_loaded (str): Specifies which model is currently loaded ('hide' or 'reveal').
    """
    def __init__(self, 
                 model_path_hide: str,
                 model_path_reveal: str,
                 framework: str, 
                 device: str = 'cpu',
                 custom_objects: Optional[Dict] = None,
                 objective: str = 'hide',
                 post_process_hide_func: Optional[Callable] = None,
                 pre_process_reveal_func: Optional[Callable] = None):
        
        self.objetive = objective
        self.device = device
        self.framework = framework
        self.custom_objects = custom_objects
        self.model_path_hide = model_path_hide
        self.model_path_reveal = model_path_reveal
        self.post_process_hide_func = post_process_hide_func
        self.pre_process_reveal_func = pre_process_reveal_func
        if self.objetive == 'both' or self.objetive == 'hide':
            # First load the hide model
            self.model = self.load_model(self.model_path_hide)
            self.model_loaded = 'hide'
        else:
            # Load reveal model
            self.model = self.load_model(self.model_path_reveal)
            self.model_loaded = 'reveal'
    
    def load_model(self, model_path):
        # load the model
        if self.framework == 'tensorflow':
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, 
                                               custom_objects=self.custom_objects)
            
            return model
        elif self.framework == 'pytorch':
            import torch
            model = torch.load(model_path, map_location=self.device)
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
        if self.model_loaded == 'reveal':
            self.gpu_memory_reset()
            self.model = self.load_model(self.model_path_hide)
            self.model_loaded = 'hide'
        
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
    
    def reveal(self, stego_dict, batch_size=8):
        """
        Reveals a secret image from stego images using the pre-trained model."
        """
        if self.model_loaded == 'hide':
            self.gpu_memory_reset()
            self.model = self.load_model(self.model_path_reveal)
            self.model_loaded = 'reveal'

        # if a pre-processing function was provided, apply it.
        if self.pre_process_reveal_func is not None:
            pre_processed_stegos = self.pre_process_reveal_func(stego_dict)

        num_images = len(pre_processed_stegos)
        
        batch_padding = (batch_size - (num_images % batch_size)) % batch_size
        if batch_padding > 0:
            filler = np.zeros((batch_padding,) + pre_processed_stegos.shape[1:], 
                              dtype=pre_processed_stegos.dtype)
            pre_processed_stegos = np.concatenate((pre_processed_stegos, filler), 
                                                  axis=0)

        if self.framework == 'tensorflow':
            secrets = self.model.predict(pre_processed_stegos, batch_size=batch_size, 
                                         verbose=True)
        elif self.framework == 'pytorch':
            with torch.no_grad():
                stegos_torch = torch.from_numpy(pre_processed_stegos).to(self.device)
                secrets = self.model(input=stegos_torch)
                secrets = secrets.cpu().numpy()

        # You may clip the values to be able to use the result as an image 
        norm_recovered_secret_arr = np.clip(secrets[:num_images], 0 , 1)
        
        return {"secret_images": norm_recovered_secret_arr}
    
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
        
    