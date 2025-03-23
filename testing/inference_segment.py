#----------------------------------------------------------------------------------
# THIS PROGRAM TAKES A TEST CSV FILE WITH ABSOLUTE NIFTI IMAGE PATH AND PASS
# THROUGH TRAINED MODELS TO GENERATE SEGMENTATION MASK AND SEGMENTED MIP IMAGE
#----------------------------------------------------------------------------------
""" Note: Here, Input file is substracted T1_Fat_Sat Nifti file """
"""       a) User need to convert nifti image of Axial subtracted T1_Fat_Sat (i.e., T1_post1-T1_pre) prior to implementing the code 
          b) The program works well for tensorflow version 2.5.3 since model uses lamda layer which is depricated 
             in recent versions of tensorflow
          c) Download the trained weight from https://drive.google.com/drive/folders/1Xdjpgld-xnEfAy1f1iTcq6BnfnibKODu?usp=sharing
             and copy weights from download folder to output folder """
#
"""IMPORT ESSENTIAL LIBRARIES"""
import os, sys, logging, time
logging.disable(logging.WARNING)  
# Hides warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow           as tf
import numpy                as np
import pandas               as pd
import nibabel              as nib
from scipy                  import ndimage
import matplotlib.pyplot    as plt
#
""" Load custom functions """
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from support_function import custom_model_unet
#
#######
""" STEP - 1: SOME FUNCTIONS CALLED IN MAIN """
######
""" (A) Function to convert T1_FatSat_substracted NIFTI image 
    (= T1_FatSat_PostContrast 1 - T1_FatSat_PreContrast) to mip and three sub-mip """
def load_data(imgpaths_temp):
    """Load image
       :param imgpaths_temp: dummy variable for image path
       :return: mip, submip13, submip23, submip33
    """
    #
    """ Read image """
    image = nib.load(os.path.join(imgpaths_temp)).get_fdata() # shape (w, h, slice)
    #
    if (image.shape[0] == 512) and (image.shape[1] == 512):
        pass
    else:
        width_factor = 512/image.shape[0]
        height_factor = 512/image.shape[1]
        depth_factor = 1
        # RESIZE 3D IMAGE MATRIX USING SPLINE INTERPOLATION OF THE ORDER
        image = ndimage.zoom(image, (width_factor, height_factor, depth_factor), order=1)
    #
    # Extract submips from nifti data (it assumes nifti array shape (w, h, slice_num)
    length  = int(image.shape[2]/3) # Divide the slice number of 3D image by 3
    mip     = np.amax(image[:, :, 0:image.shape[2]],2)
    submip13=np.amax(image[:, :, 0:length],2)
    submip23=np.amax(image[:, :, length:(2*length)],2)
    submip33=np.amax(image[:, :, (2*length):image.shape[2]],2)
    # Rescale the image from 0 to 255 (unet has lamda layer that is rescaling inside)
    mip      = ((mip/(np.amax(mip)-np.amin(mip)))*255).clip(0,255)
    submip13 = ((submip13/(np.amax(submip13)-np.amin(submip13)))*255).clip(0,255)
    submip23 = ((submip23/(np.amax(submip23)-np.amin(submip23)))*255).clip(0,255)
    submip33 = ((submip33/(np.amax(submip33)-np.amin(submip33)))*255).clip(0,255)
    # Add an axis to sub mips
    mip      = (mip[..., tf.newaxis]).astype("float32") # (512, 512, 1)
    submip13 = (submip13[..., tf.newaxis]).astype("float32") # (512, 512, 1)
    submip23 = (submip23[..., tf.newaxis]).astype("float32") # (512, 512, 1)
    submip33 = (submip33[..., tf.newaxis]).astype("float32") # (512, 512, 1)
    #
    return mip, submip13, submip23, submip33
#
""" (B) Function to extract the images """
def process_data(x):
    #
    """This function preprocess the input data for tf.data API
        :param x: dummy variables for image path
        :return: (mip, submip13, submip23, submip33)
    """
    #
    def f(x):
        #
        path = x.decode()     # decodes the tf string to readable string
        mip, submip13, submip23, submip33 = load_data(path)
        #
        return mip, submip13, submip23, submip33
    #
    mip, submip13, submip23, submip33 = tf.numpy_function(f, [x], [tf.float32, tf.float32, tf.float32, tf.float32])
    #
    mip.set_shape([512, 512, 1])
    submip13.set_shape([512, 512, 1])
    submip23.set_shape([512, 512, 1])
    submip33.set_shape([512, 512, 1])
    #
    return ((mip, submip13, submip23, submip33), )
#
""" (C) CONVERT INTO TF DATA API PIPELINE dataset iterable
        Function to create a tf_dataset
"""
def tf_dataset(x, func_name, batch):
    #
    """This function tf_dataset
        :param x: dummy variables for path
        :param func_name: dummy variables for calling function
        :param batch: dummy variables for batch data generation
        :return: tf_dataset
    """
    #
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.map(func_name)
    dataset = dataset.batch(batch)
    return dataset
#
#
""" (D) Function for loading the encoders """
def load_unet(dirpath, filename):
    #
    model=custom_model_unet.unet(224, 1)
    model.load_weights(os.path.join(dirpath, filename))
    #
    return model
#
""" (E) Function to combine Unet prediction (output mask and segmented image)"""
def create_mask(function, dirpath, filename):
    #
    input1 = tf.keras.Input((512, 512, 1)) 
    # Input shape of image (None, 512, 512, 1)
    x = tf.keras.layers.Resizing(224, 224, interpolation='bilinear')(input1) 
    # Resized the input to input shape of unet (None, 224, 224, 1)
    segmenter  = function(dirpath, filename)
    segmenter.trainable = False
    # load the pre-trained vgg16 model
    x = segmenter(x)
    # Extract the mask semented from thorax shape (none, 224, 224, 1)
    x = tf.cast(tf.math.greater_equal(x, 0.5), tf.float32)
    # Extract the binary value (0 and 1) of shape (none, 224, 224, 1)
    x_mask = tf.keras.layers.Resizing(512, 512, interpolation='bilinear')(x)
    # Resize the binary mask value to input image shape (none, 512, 512, 1)
    #
    model = tf.keras.models.Model(inputs = [input1], outputs=[x_mask], name = "unet_enconder")
    #
    return model
#
""" (F) Function to combine multiple input"""
def combined_input(function1, function2, dirpath, filename):
    #
    input1 = tf.keras.Input((512, 512, 1))
    input2 = tf.keras.Input((512, 512, 1))
    input3 = tf.keras.Input((512, 512, 1))
    input4 = tf.keras.Input((512, 512, 1))
    #
    encoder = function2(function1, dirpath, filename)
    out1 = encoder(input1)
    out2 = encoder(input2)
    out3 = encoder(input3)
    out4 = encoder(input4)
    #
    model = tf.keras.models.Model(inputs = [input1, input2, input3, input4], outputs=[out1, out2, out3, out4], name = "Ensemble_segmenter")
    #
    return model
#
""" (G) Function to fill holes in binary image and save plot """
def fill_holes(mask):
    mask=ndimage.binary_fill_holes(mask).astype(np.float32) 
    # Fills the holes in the binary mask
    return mask
#
def save_plot(save_path, im):
    #
    plt.figure(1);
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False);
    plt.imshow(im, cmap='gray');
    plt.imsave(save_path, im, cmap='gray');  # saves the file to path
    plt.clf();
#
#######
""" STEP - 2: PREDICTION """
######
#
def ai_predict_fn(batch, default_path, file_path, file_name, img_type):
    #
    """ Input hyperparameters """
    hp = {}
    hp["file_path"]      = file_path                            # file path
    hp["file_name"]      = file_name                            # csv or nifti filename
    hp["model_path"]     = os.path.join(default_path, 'output') # model path
    hp["saved_model"]    = "unet_weight.h5"                     # unet weight
    hp["batch_size"]     = batch                                # batch size
    hp["plot_img"]       = img_type                             # plot mip or sub-mips
    #
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    #
    """ Checks availability of gpu """
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available:
        """ Set gpu device """
        tf.config.set_visible_devices(gpu_available[0], 'GPU')
    #
    """ load combined input models (unet multi-input mip, submip13-submip33) """
    model = combined_input(load_unet, create_mask, hp["model_path"], hp["saved_model"])
    #
    """ load the list of images or single image """
    if hp["file_name"].endswith('.csv'):
        df = pd.read_csv(os.path.join(hp["file_path"], hp["file_name"]))
        img_paths  = df.File_path.tolist()
    elif hp["file_name"].endswith('.xlsx'):
        df = pd.read_excel(os.path.join(hp["file_path"], hp["file_name"]))
        img_paths  = df.File_path.tolist()
    #
    """ loading tf.dataset data structure """
    ds_test= tf_dataset(img_paths, process_data, hp["batch_size"])
    #
    """ Prediction for each accession """
    pred = model.predict(ds_test, verbose = 0)
    #
    """ INTERPRETATION OF TUPLE pred variables
        prediction output tuple shape (4, batch, 512, 512, 1) such that (1st, 2nd, 3rd, 4th, 5th) refers to
        1st position -> type of img 0-3, i.e., mip, submip13...submip33
        2nd position -> total number of exams in excel or csv
        3rd position -> image or mask w = 512
        4th position -> image or mask h = 512
        5th position -> channel
    """
    #
    for idx in range(len(img_paths)):
        #
        """ collect mask for mip, submip13-33 images """
        mip_mask, submip13_mask, submip23_mask, submip33_mask = pred[0][idx], pred[1][idx], pred[2][idx], pred[3][idx] 
        # each having shape 512, 512, 1
        #
        """ collect original images for mip, submip13-33 images (no segmentation) """
        mip, submip13, submip23, submip33 = load_data(img_paths[idx]) 
        # each having shape 512, 512, 1
        #
        # plot files
        if hp["plot_img"] == "MIP & mask":
            #
            mip_mask = fill_holes(np.squeeze(mip_mask, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "MIP.jpg"), np.squeeze(mip, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "MIP_mask.jpg"), mip_mask)
            #
        elif hp["plot_img"] == "Sub-MIP1-3 & mask":
            #
            submip13_mask = fill_holes(np.squeeze(submip13_mask, axis=-1))
            submip23_mask = fill_holes(np.squeeze(submip23_mask, axis=-1))
            submip33_mask = fill_holes(np.squeeze(submip33_mask, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_1of3.jpg"), np.squeeze(submip13, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_2of3.jpg"), np.squeeze(submip23, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_3of3.jpg"), np.squeeze(submip33, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_1of3_mask.jpg"), submip13_mask)
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_2of3_mask.jpg"), submip23_mask)
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_3of3_mask.jpg"), submip33_mask)
            #
        else:
            mip_mask = fill_holes(np.squeeze(mip_mask, axis=-1))
            submip13_mask = fill_holes(np.squeeze(submip13_mask, axis=-1))
            submip23_mask = fill_holes(np.squeeze(submip23_mask, axis=-1))
            submip33_mask = fill_holes(np.squeeze(submip33_mask, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "MIP.jpg"), np.squeeze(mip, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_1of3.jpg"), np.squeeze(submip13, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_2of3.jpg"), np.squeeze(submip23, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_3of3.jpg"), np.squeeze(submip33, axis=-1))
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "MIP_mask.jpg"), mip_mask)
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_1of3_mask.jpg"), submip13_mask)
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_2of3_mask.jpg"), submip23_mask)
            save_plot(os.path.join(os.path.dirname(img_paths[idx]), "SUB_MIP_3of3_mask.jpg"), submip33_mask)
        #
    #
    return None
#
