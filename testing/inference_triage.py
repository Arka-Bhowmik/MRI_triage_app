#----------------------------------------------------------------------------------
# THIS PROGRAM TAKES A TEST CSV FILE WITH ABSOLUTE NIFTI IMAGE PATH AND PASS
# THROUGH TRAINED MODELS TO GENERATE A PROBABILITY OUTPUT FILE AND ROC PLOT
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
#
""" Load custom functions """
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from support_function import custom_model_unet, tqdm_predict_callback
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
def load_models(dirpath, filename):
    #
    if (filename.split('_')[0]) == "unet":
        model=custom_model_unet.unet(224, 1)
        model.load_weights(os.path.join(dirpath, filename))
    else:
        model=tf.keras.models.load_model(os.path.join(dirpath, filename))
    #
    return model
#
#
""" (E) Function to combine Unet and VGG-16 encoders in series"""
def combined_encoder(function, dirpath, filename_1, filename_2):
    #
    input1 = tf.keras.Input((512, 512, 1)) 
    # Input shape of image (None, 512, 512, 1)
    x = tf.keras.layers.Resizing(224, 224, interpolation='bilinear')(input1) 
    # Resized the input to input shape of unet (None, 224, 224, 1)
    segmenter  = function(dirpath, filename_1)
    segmenter.trainable = False
    # load the pre-trained Vgg16 model
    classifier = function(dirpath, filename_2)
    classifier.trainable = False
    # load the pre-trained vgg16 model
    x = segmenter(x)
    # Extract the mask semented from thorax shape (none, 224, 224, 1)
    x = tf.cast(tf.math.greater_equal(x, 0.5), tf.float32)
    # Extract the binary value (0 and 1) of shape (none, 224, 224, 1)
    x = tf.keras.layers.Resizing(512, 512, interpolation='bilinear')(x)
    # Resize the binary mask value to input image shape (none, 512, 512, 1)
    x = tf.math.multiply(input1, x)
    # multiply the binary mask with input image shape (none, 512, 512, 1) segmented from thorax
    x = tf.keras.layers.Reshape([512, 512])(x)
    # reshape the array to shape (none, 512, 512)
    x_r = x[:,int(0):int(x.shape[2]), int(0):int(x.shape[1]/2)] # right breast
    x_l = x[:,int(0):int(x.shape[2]), int(x.shape[1]/2):int(x.shape[1])] # left breast
    # Divide into right and left breast each having shape (none, 512, 256) 
    x_r = tf.stack((x_r,)*3, -1)
    x_l = tf.stack((x_l,)*3, -1)
    # Stack the input channel three times each having shape (none, 512, 256, 3)
    x_r = tf.keras.layers.Resizing(256, 256, interpolation='bilinear')(x_r)
    x_l = tf.keras.layers.Resizing(256, 256, interpolation='bilinear')(x_l)
    # Resize the shape of image to input shape of vgg-16 (none, 256, 256, 3)
    x_r = tf.keras.layers.Lambda(lambda x: (x) / (tf.math.reduce_max(x)))(x_r)
    x_l = tf.keras.layers.Lambda(lambda x: (x) / (tf.math.reduce_max(x)))(x_l)
    # Normalize the images from 0 to 1
    out_r = classifier(x_r)
    out_l = classifier(x_l)
    # Vgg-16 output shape (None, 2)
    model = tf.keras.models.Model(inputs = [input1], outputs=[out_r, out_l], name = "combined_enconder")
    #
    return model
#
#
""" (F) Function to combine multiple input """
def combined_model(function1, function2, dirpath, filename_1, filename_2):
    #
    input1 = tf.keras.Input((512, 512, 1))
    input2 = tf.keras.Input((512, 512, 1))
    input3 = tf.keras.Input((512, 512, 1))
    input4 = tf.keras.Input((512, 512, 1))
    #
    encoder = function2(function1, dirpath, filename_1, filename_2)
    out1 = encoder(input1)
    out2 = encoder(input2)
    out3 = encoder(input3)
    out4 = encoder(input4)
    #
    model = tf.keras.models.Model(inputs = [input1, input2, input3, input4], outputs=[out1, out2, out3, out4], name = "Ensemble_model")
    #
    return model
#
#######
""" STEP - 2: PREDICTION """
######
#
def ai_predict_fn(batch, threshold, default_path, file_path, file_name):
    #
    """ Input hyperparameters """
    hp = {}
    hp["file_path"]      = file_path                            # file path
    hp["file_name"]      = file_name                            # CSV or nifti filename
    hp["model_path"]     = os.path.join(default_path, 'output') # Model path
    hp["saved_model"]    = ["unet_weight.h5", "vgg16_weight_fold1.h5", "vgg16_weight_fold2.h5", 
                            "vgg16_weight_fold3.h5", "vgg16_weight_fold4.h5", "vgg16_weight_fold5.h5"]
    hp["batch_size"]     = batch
    hp["test_threshold"] = threshold                            # algorithm threshold
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
    """ load combined models (unet + fold1-5) """
    model_fold1 = combined_model(load_models, combined_encoder, hp["model_path"], hp["saved_model"][0], hp["saved_model"][1])
    model_fold2 = combined_model(load_models, combined_encoder, hp["model_path"], hp["saved_model"][0], hp["saved_model"][2])
    model_fold3 = combined_model(load_models, combined_encoder, hp["model_path"], hp["saved_model"][0], hp["saved_model"][3])
    model_fold4 = combined_model(load_models, combined_encoder, hp["model_path"], hp["saved_model"][0], hp["saved_model"][4])
    model_fold5 = combined_model(load_models, combined_encoder, hp["model_path"], hp["saved_model"][0], hp["saved_model"][5])
    #
    """ load the list of images or single image """
    if hp["file_name"].endswith('.csv'):
        df = pd.read_csv(os.path.join(hp["file_path"], hp["file_name"]))
        img_paths  = df.File_path.tolist()
        img_labels = df.Patho.tolist()
        img_mrns = df.MRN.tolist()
        img_accs = df.Accession.tolist()
    elif hp["file_name"].endswith('.xlsx'):
        df = pd.read_excel(os.path.join(hp["file_path"], hp["file_name"]))
        img_paths  = df.File_path.tolist()
        img_labels = df.Patho.tolist()
        img_mrns = df.MRN.tolist()
        img_accs = df.Accession.tolist()
    else:
        img_paths = [os.path.join(hp["file_path"], hp["file_name"])] # save as a list
    #
    #
    """ loading tf.dataset data structure """
    ds_test= tf_dataset(img_paths, process_data, hp["batch_size"])
    #
    """ Prediction for each accession """
    print("Cross-validation model predictions:")
    time1 = time.time()
    pred_fold1 = model_fold1.predict(ds_test, callbacks=[tqdm_predict_callback.TQDMPredictCallback()], verbose = 0) # fold1
    time2 = time.time()
    print("               Fold1 model prediction completed (CPU/GPU time = ", round((time2-time1)/60, 2), " min)")
    pred_fold2 = model_fold2.predict(ds_test, callbacks=[tqdm_predict_callback.TQDMPredictCallback()], verbose = 0) # fold2
    time3 = time.time()
    print("               Fold2 model prediction completed (CPU/GPU time = ", round((time3-time2)/60, 2), " min)")
    pred_fold3 = model_fold3.predict(ds_test, callbacks=[tqdm_predict_callback.TQDMPredictCallback()], verbose = 0) # fold3
    time4 = time.time()
    print("               Fold3 model prediction completed (CPU/GPU time = ", round((time4-time3)/60, 2), " min)")
    pred_fold4 = model_fold4.predict(ds_test, callbacks=[tqdm_predict_callback.TQDMPredictCallback()], verbose = 0) # fold4
    time5 = time.time()
    print("               Fold4 model prediction completed (CPU/GPU time = ", round((time5-time4)/60, 2), " min)")
    pred_fold5 = model_fold5.predict(ds_test, callbacks=[tqdm_predict_callback.TQDMPredictCallback()], verbose = 0) # fold5
    time6 = time.time()
    print("               Fold5 model prediction completed (CPU/GPU time = ", round((time6-time5)/60, 2), " min)")
    #
    """ INTERPRETATION OF TUPLE pred_fold1 - pred_fold5 variables
        prediction output tuple shape (4, 2, 4, 2) such that (1st, 2nd, 3rd, 4th) refers to
        1st position -> type of img 0-3, i.e., mip, submip13...submip33
        2nd position -> img laterality 0-1, i.e., 0 being right and 1 being left
        3rd position -> total number of exams in excel or csv
        4th position -> probability score 0-1, i.e., position 0 being negative probability and position 1 being positive probability
    """
    data=[]
    #
    for idx in range(len(img_paths)):
        #
        """ average of score for mip images """
        prediction_right_neg=(pred_fold1[0][0][idx][0]+pred_fold2[0][0][idx][0]+pred_fold3[0][0][idx][0]+pred_fold4[0][0][idx][0]+pred_fold5[0][0][idx][0])/5
        prediction_right_pos=(pred_fold1[0][0][idx][1]+pred_fold2[0][0][idx][1]+pred_fold3[0][0][idx][1]+pred_fold4[0][0][idx][1]+pred_fold5[0][0][idx][1])/5
        prediction_left_neg=(pred_fold1[0][1][idx][0]+pred_fold2[0][1][idx][0]+pred_fold3[0][1][idx][0]+pred_fold4[0][1][idx][0]+pred_fold5[0][1][idx][0])/5
        prediction_left_pos=(pred_fold1[0][1][idx][1]+pred_fold2[0][1][idx][1]+pred_fold3[0][1][idx][1]+pred_fold4[0][1][idx][1]+pred_fold5[0][1][idx][1])/5
        #
        """ average of score for submip 13 """
        prediction_right_1of3_neg=(pred_fold1[1][0][idx][0]+pred_fold2[1][0][idx][0]+pred_fold3[1][0][idx][0]+pred_fold4[1][0][idx][0]+pred_fold5[1][0][idx][0])/5
        prediction_right_1of3_pos=(pred_fold1[1][0][idx][1]+pred_fold2[1][0][idx][1]+pred_fold3[1][0][idx][1]+pred_fold4[1][0][idx][1]+pred_fold5[1][0][idx][1])/5
        prediction_left_1of3_neg=(pred_fold1[1][1][idx][0]+pred_fold2[1][1][idx][0]+pred_fold3[1][1][idx][0]+pred_fold4[1][1][idx][0]+pred_fold5[1][1][idx][0])/5
        prediction_left_1of3_pos=(pred_fold1[1][1][idx][1]+pred_fold2[1][1][idx][1]+pred_fold3[1][1][idx][1]+pred_fold4[1][1][idx][1]+pred_fold5[1][1][idx][1])/5
        #
        """ average of score for submip 23 """
        prediction_right_2of3_neg=(pred_fold1[2][0][idx][0]+pred_fold2[2][0][idx][0]+pred_fold3[2][0][idx][0]+pred_fold4[2][0][idx][0]+pred_fold5[2][0][idx][0])/5
        prediction_right_2of3_pos=(pred_fold1[2][0][idx][1]+pred_fold2[2][0][idx][1]+pred_fold3[2][0][idx][1]+pred_fold4[2][0][idx][1]+pred_fold5[2][0][idx][1])/5
        prediction_left_2of3_neg=(pred_fold1[2][1][idx][0]+pred_fold2[2][1][idx][0]+pred_fold3[2][1][idx][0]+pred_fold4[2][1][idx][0]+pred_fold5[2][1][idx][0])/5
        prediction_left_2of3_pos=(pred_fold1[2][1][idx][1]+pred_fold2[2][1][idx][1]+pred_fold3[2][1][idx][1]+pred_fold4[2][1][idx][1]+pred_fold5[2][1][idx][1])/5
        #
        """ average of score for submip 33 """
        prediction_right_3of3_neg=(pred_fold1[3][0][idx][0]+pred_fold2[3][0][idx][0]+pred_fold3[3][0][idx][0]+pred_fold4[3][0][idx][0]+pred_fold5[3][0][idx][0])/5
        prediction_right_3of3_pos=(pred_fold1[3][0][idx][1]+pred_fold2[3][0][idx][1]+pred_fold3[3][0][idx][1]+pred_fold4[3][0][idx][1]+pred_fold5[3][0][idx][1])/5
        prediction_left_3of3_neg=(pred_fold1[3][1][idx][0]+pred_fold2[3][1][idx][0]+pred_fold3[3][1][idx][0]+pred_fold4[3][1][idx][0]+pred_fold5[3][1][idx][0])/5
        prediction_left_3of3_pos=(pred_fold1[3][1][idx][1]+pred_fold2[3][1][idx][1]+pred_fold3[3][1][idx][1]+pred_fold4[3][1][idx][1]+pred_fold5[3][1][idx][1])/5
        #
        ensemble_probability = max(prediction_right_pos, prediction_left_pos, prediction_right_1of3_pos, 
                                   prediction_right_2of3_pos, prediction_right_3of3_pos, prediction_left_1of3_pos,
                                   prediction_left_2of3_pos, prediction_left_3of3_pos)
        """ Class prediction """
        if (prediction_right_pos<hp["test_threshold"]) and (prediction_left_pos<hp["test_threshold"]) and (prediction_right_1of3_pos<hp["test_threshold"]) and (prediction_right_2of3_pos<hp["test_threshold"]) and (prediction_right_3of3_pos<hp["test_threshold"]) and (prediction_left_1of3_pos<hp["test_threshold"]) and (prediction_left_2of3_pos<hp["test_threshold"]) and (prediction_left_3of3_pos<hp["test_threshold"]):
            pred_labels=0
        else:
            pred_labels=1
        #
        #
        """ Store data """
        if ((hp["file_name"].endswith('.csv')) or (hp["file_name"].endswith('.xlsx'))):
            data.append((img_mrns[idx], img_accs[idx], img_paths[idx], prediction_right_pos, prediction_right_1of3_pos, 
                         prediction_right_2of3_pos, prediction_right_3of3_pos, prediction_left_pos, prediction_left_1of3_pos, 
                         prediction_left_2of3_pos, prediction_left_3of3_pos, ensemble_probability, img_labels[idx], pred_labels))
            #
        else:
            data.append((img_paths[idx], prediction_right_pos, prediction_right_1of3_pos, 
                         prediction_right_2of3_pos, prediction_right_3of3_pos, prediction_left_pos, prediction_left_1of3_pos, 
                         prediction_left_2of3_pos, prediction_left_3of3_pos, ensemble_probability, pred_labels))
            #
        #
    #
    """ header names """
    col1=['MRN', 'Accession', 'File_path', 'Prob_right_mip', 'Prob_right13_submip', 'Prob_right23_submip', 
          'Prob_right33_submip', 'Prob_left_mip', 'Prob_left13_submip', 'Prob_left23_submip', 
          'Prob_left33_submip','Ensemble_probability', 'Ground_labels', 'Predicted_labels']
    col2=['File_path', 'Prob_right_mip', 'Prob_right13_submip', 'Prob_right23_submip', 
          'Prob_right33_submip', 'Prob_left_mip', 'Prob_left13_submip', 'Prob_left23_submip', 
          'Prob_left33_submip','Ensemble_probability', 'Predicted_labels']
    #
    if ((hp["file_name"].endswith('.csv')) or (hp["file_name"].endswith('.xlsx'))):
        output_table = pd.DataFrame(data, columns=col1)
    else:
        output_table = pd.DataFrame(data, columns=col2)
    #
    output_table.to_csv(os.path.join(hp["model_path"], 'probability.csv'), index=False)    # SAVES THE OUTPUT TABLE
    #
    return output_table
