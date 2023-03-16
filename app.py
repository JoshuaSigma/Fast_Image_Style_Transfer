import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import streamlit as st

# Fx's ---------------------------------------------------------------------------------
def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

# Decs -------------------------------------------------------------------------------------
content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'  # @param {type:"string"}
output_image_size = 384  # @param {type:"integer"}

# The content image size can be arbitrary.
content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the 
# recommended image size for the style image (though, other sizes work as 
# well but will lead to different results).
style_img_size = (256, 256)  # Recommended to keep it at 256.

content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

# Load TF Hub module.
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Code --------------------------------------------------------------------------------------------
fxs = '''def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img
'''

example = '''content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'
style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'

output_image_size = 384
content_img_size = (output_image_size, output_image_size)

style_img_size = (256, 256)  # Recommended to keep it at 256.

content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
'''

magic = '''hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Stylize content image with given style image
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]
'''

imports = '''import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
'''

# Document Render ----------------------------------------------------------------------------------------

# Title
st.title("Fast style transfer performed locally")
st.text("Joshua Patterson | March 15, 2023")

with st.container():
    # Columns of example images
    col3, col4, col5 = st.columns(3)

    st.caption("Based on the model code in magenta and the publication: Exploring the structure of a real-time, arbitrary neural artistic stylization network. Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens, Proceedings of the British Machine Vision Conference (BMVC), 2017.")

with st.container():
    st.header("Transfer the style of one photo to another")
    st.write("Style Transfering involves taking the style of one image, usually a painting or digital art piece, and transferring that style to a target photo. This model was trained on approximately 80,000 paintings and 6,000 textures. And while it is limited to one style, it performs in real-time.")

with st.container():
    st.header("On device style transfer")
    st.write("The first and second images above are normal downloaded images. However, the last image is produced using magenta in real-time on this device. Code and upload tester below. Style image works best at 256 x 256.")

with st.container():
    st.header("Import Packages")
    st.write("We start by importing the necessary packages. Make sure your environment has these dependencies installed. ")
    st.code(imports, language='python')

with st.container():
    st.write("Next, Lets define our preprocessing functions.")
    st.code(fxs, language='python')

with st.container():
    st.write("Then we grab some images from the web. The image that is being altered, content image, can be of arbitrary size. Since the model was trained on 256 x 256 images, it works best with 256. We are also going to add an average pooling layer that you will be able to play around with below. For now its set at Kernel size = [3,3] and stride = [1,1]")
    st.code(example, language='python')

with st.container():
    st.write("Now for the magic. We simply load the model and process the output image.")
    st.code(magic, language='python')

st.header("Let's try it out!")
st.write("Add a picture you want altered on the left (top if on small screen), then add a style image to get a stylized image below and a couple of sliders to change the pooling layer. The style image is designed to work at 256 x 256, but should take any image size.")

# Stylize content image with given style image.
# This is pretty fast within a few milliseconds on a GPU.

outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Visualize input images and the generated stylized image.
col3.image(content_image.numpy())
col4.image(style_image.numpy())
col5.image(stylized_image.numpy())

imga = []
imgb = []
col6, col7 = st.columns(2)
custom_image_to_alter = col6.file_uploader(label="Content image to recieve style", type=['png', 'jpg', 'jpeg'])
if custom_image_to_alter is not None:
    imga = tf.io.decode_image(
    custom_image_to_alter.read(),
    channels=3, dtype=tf.float32)[tf.newaxis, ...]
    imga = crop_center(imga)
    imga = tf.image.resize(imga, (256, 256), preserve_aspect_ratio=True)
    print(imga.shape)
    col6.image(custom_image_to_alter)


style_image_custom = col7.file_uploader(label="Style image to give style", type=['png', 'jpg'])
if style_image_custom is not None:
    imgb = tf.io.decode_image(
    style_image_custom.read(),
    channels=3, dtype=tf.float32)[tf.newaxis, ...]
    imgb = crop_center(imgb)
    imgb = tf.image.resize(imgb, (256, 256), preserve_aspect_ratio=True)
    print(imgb.shape)
    col7.image(style_image_custom)

if style_image_custom and custom_image_to_alter is not None:
    col8, col9 = st.columns(2)
    kernel_size = col8.slider('Kernel Size', 1, 10, 3)
    stride_size = col9.slider('Stride', 1, 3, 1)

    imgb = tf.nn.avg_pool(imgb, ksize=[kernel_size, kernel_size], strides=[stride_size, stride_size], padding='SAME')

    outputs1 = hub_module(imga, imgb)
    stylized_image1 = outputs1[0]

    st.image(stylized_image1.numpy())
