import numpy as np
import tensorflow as tf
import tensorflow_wavelets.Layers.DWT as DWT

##### Custom layers ####################################################
# 1) Custom layer that does the Discrete Wavelet Transform of an image.
# Used in Steguz model.
def get_IDWT_TF(scale):
	"""
    Returns a model that performs the Inverse Discrete Wavelet Transform (IDWT) of an image."""
    # Define the input tensor
	tensor = tf.keras.layers.Input(shape = (int(scale/2), 
													   int(scale/2), 4, 3), 
													   dtype='float64', batch_size = 1)

	# Concatenate each of the approximation and detail images of the DWT, for each color channel
	channels = []
	for channel in range(3):
		tensor_channel = tensor[:,:,:,:,channel]
		ll = tf.reshape(tensor_channel[:,:,:,0], (1, int(scale/2), int(scale/2)))
		lh = tf.reshape(tensor_channel[:,:,:,1], (1, int(scale/2), int(scale/2)))
		hl = tf.reshape(tensor_channel[:,:,:,2], (1, int(scale/2), int(scale/2)))
		hh = tf.reshape(tensor_channel[:,:,:,3], (1, int(scale/2), int(scale/2)))

		channels.append(tf.reshape(tf.concat([tf.concat([ll, lh], axis=1), 
										tf.concat([hl, hh], axis=1)], axis=2), (1, scale, scale)))

	tmp = tf.convert_to_tensor(channels)
	concat_tensor = tf.stack([tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:]], axis=-1)

	# IDWT layers on each color channel
	scope_name = 'hide_IDWT'
	idwt_tensor_0 = DWT.IDWT(name = scope_name + "0", splited = 0)(tf.reshape(concat_tensor[:,:,:,0], (1, scale, scale, 1), 'reshape_channel_0'))
	idwt_tensor_1 = DWT.IDWT(name = scope_name + "1", splited = 0)(tf.reshape(concat_tensor[:,:,:,1], (1, scale, scale, 1), 'reshape_channel_1'))
	idwt_tensor_2 = DWT.IDWT(name = scope_name + "2", splited = 0)(tf.reshape(concat_tensor[:,:,:,2], (1, scale, scale, 1), 'reshape_channel_2'))

	# And then we stack the channels and reshape to yield a image with dimensions (1, scale, scale, 3).
	tensor_stacked = tf.stack([idwt_tensor_0, idwt_tensor_1, idwt_tensor_2], axis=-1, name='hide_stack')
	tensor_idwt = tf.reshape(tensor_stacked, (1, scale, scale, 3), name = 'final_reshape_idwt')

	# Define the model
	IDWT_net = tf.keras.Model(
        inputs=[tensor],
        outputs=[tensor_idwt],
        name = 'IDWT_NET',
	)

	return IDWT_net
def get_DWT_TF(scale):
	"""
	Returns a model that performs the Discrete Wavelet Transform (DWT) of an image."""
	# Define the input tensor
	tensor = tf.keras.layers.Input(shape = (int(scale), int(scale), 3), 
								dtype='float64', batch_size = 1)

	# Compute the DWT of each of the channels of an RGB image.
	scope_name = 'DWT'
	dwt_0 = DWT.DWT(name=scope_name + "0",concat=0)(tf.reshape(tensor[:,:,:,0], (1, scale, scale, 1)))
	dwt_1 = DWT.DWT(name=scope_name + "1",concat=0)(tf.reshape(tensor[:,:,:,1], (1, scale, scale, 1)))
	dwt_2 = DWT.DWT(name=scope_name + "2",concat=0)(tf.reshape(tensor[:,:,:,2], (1, scale, scale, 1)))

	# We then stack all of the DWTs of each of the channels into one array
	tensor_stacked = tf.stack([dwt_0, dwt_1, dwt_2], axis = -1)

	# Define the model
	DWT_net = tf.keras.Model(
	inputs=[tensor],
	outputs=[tensor_stacked],
	name = 'DWT_net',
	)

	return DWT_net

## Custom loss functions ###############################################
# 1) Loss function used in the Steguz model. Required to load the model
def get_steguz_loss(y_true, y_pred):
	gain = 0
	gain += tf.image.psnr(y_true, y_pred, 1.)
	gain += tf.image.ssim(y_true, y_pred, 1.)

	return -1*gain

# Custom post processing hide functions ################################
# 1) Post-processing function used for the Steguz model.
def post_process_hide_func_steguz(stego_images, scale):
    """
    Post-process the stego images obtained from the Steguz model.
    Args:
        stego_images (np.ndarray): Stego images obtained from the Steguz model.
        scale (int): Dim of the image.
    Returns: dict
        stego_images (np.ndarray): Post-processed stego images.
		min_values (list): Minimum values of the normalized stego images.
		max_values (list): Maximum values of the normalized stego images
    """
	
    # Create the model that performs the IDWT on a image
    idwt_net = get_IDWT_TF(scale)
	# Perform the IDWT on the stego images in a loop
    min_values = []
    max_values = []
    stego_images_full_resolution = []
    for i in range(len(stego_images)):
        idwt_stego = idwt_net.predict(stego_images[i].reshape(1, int(scale/2), 
																   int(scale/2),
																	4, 3))
        # Reshape the IDWT output to have the same shape as the original image
        stego_arr = np.reshape(idwt_stego, (scale, scale, 3))
        # Do normalization to be able to save the as a image
        min_value = np.min(stego_arr)
        max_value = np.max(stego_arr)
        stego_norm = (stego_arr - min_value) / (max_value - min_value)
        stego_images_full_resolution.append((stego_norm * 255).astype(np.uint8))
        min_values.append(min_value)
        max_values.append(max_value)
	
    return {"stego_images": stego_images_full_resolution,
			 "min_values": min_values, 
			 "max_values": max_values}

# Custom Pre-processing reveal functions ###############################
# 1) Pre-processing reveal function used for the Steguz model.
def pre_process_reveal_func_steguz(stego_dict, scale):
	"""
	Pre-process the stego images before revealing the secret image using the Steguz model.
	Args:
		stego_images (np.ndarray): Stego images to reveal the secret image from.
		scale (int): Dim of the image.
		min_norm_values (list): Minimum values of the normalized stego images.
		max_norm_values (list): Maximum values of the normalized stego images.
	"""
	# Create the model that performs the DWT on a image
	dwt_net = get_DWT_TF(scale)
	# Perform the DWT on the stego images in a loop
	stego_images_low_resolution = []
	for i in range(len(stego_dict["stego_images"])):
		# Do the inverse normalization, and then the DWT
		stego_dict["stego_images"][i] = stego_dict["stego_images"][i].astype(np.float64)
		stego_dict["stego_images"][i] = (stego_dict["stego_images"][i] * \
								   (stego_dict["max_values"][i] - stego_dict["min_values"][i])) + stego_dict["min_values"][i]
		# Apply the DWT
		stego_dwt = dwt_net(np.reshape(stego_dict["stego_images"][i], (1, scale, scale, 3)))
		stego_images_low_resolution.append(stego_dwt)
	
	# Reshape the stego images to have the proper shape
	stego_shape = (int(scale / 2), int(scale / 2), 4, 3) 

	return np.array(stego_images_low_resolution).reshape((-1,) + stego_shape)
