#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import wget
from pathlib import Path
import joblib

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

import keras
import keras.optimizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score


DEBUG = False
if DEBUG:
    image_shape = (550, 425) #(650, 525) ## (width, height)
    max_samples = 100 #10
    num_epochs = 2 #10

else:
    image_shape = (550, 425) #(650, 525)
    max_samples = 2000 #None
    num_epochs = 100

BATCH_SIZE = 4
USE_WANDB = False
TRAIN_FOLDS = [1,2,3,4,5,6,7,8]
VAL_FOLDS = [9,10]
label_mapping = {"NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"}
LABELS = sorted(label_mapping) ## a list of labels

joblib.dump({'TRAIN_FOLDS': TRAIN_FOLDS, 'VAL_FOLDS': VAL_FOLDS}, 'folds.pkl')


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
def train_models(data_folder, model_folder, verbose):
    csv_path ='multilabel_classification.csv'
    if not os.path.exists(csv_path):
        physio_paths = list(Path(data_folder).rglob('*.hea'))
        physio_paths = [str(i) for i in physio_paths]
        df = extract_info_from_hea(label_mapping, physio_paths)

    #ptbxl_db_path = os.path.join(data_folder, 'ptbxl_database.csv')
    ptbxl_db_path = 'ptbxl_database.csv'
    if not os.path.exists(ptbxl_db_path):
        if verbose:
            print("Downloading ptbxl_database.csv...")
        wget.download('https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv', out=ptbxl_db_path)
        if verbose:
            print(f"ptbxl_database.csv downloaded successfully to {ptbxl_db_path}")

        ptbxl_df = pd.read_csv(ptbxl_db_path)
        ptbxl_df[['filename_hr', 'strat_fold']].head()

        df['strat_fold'] = df['Image_Path'].str[-31:-6].map(dict(zip(ptbxl_df['filename_hr'], ptbxl_df['strat_fold'])))
        df.head()

        df.to_csv(csv_path, index=False)
    else :
        df = pd.read_csv(csv_path)

    train_df = df[df['strat_fold'].isin(TRAIN_FOLDS)]
    val_df = df[df['strat_fold'].isin(VAL_FOLDS)]

    train_dataset = get_dataset(
      BATCH_SIZE,
      img_size = image_shape,
      df=train_df,
      max_dataset_len=max_samples,
    )

    valid_dataset = get_dataset(
      BATCH_SIZE,
      img_size = image_shape,
      df=val_df,
      max_dataset_len=max_samples,
    )

    ## use prefetching to optimize loading speed.
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE).cache()
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE).cache() ## Is this repetitive? Having already used ""num_parallel_calls=tf_data.AUTOTUNE""

    ## Save a single batch in png/pdf format
    num_images = min(BATCH_SIZE, max_samples)
    num_columns = 3
    subplot_width = 5.5
    subplot_height = 4.25
    num_rows = int(np.ceil(np.sqrt(num_images)))

    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * subplot_width, num_rows * subplot_height))
    axes = axes.flatten()

    for imgs, lbls in train_dataset.take(1):
        for i in range(num_images):
            image = imgs[i].numpy().astype('uint8')
            title = ', '.join(np.array(LABELS)[lbls[i].numpy()==1])
            axes[i].imshow(image)
            axes[i].set_title(title)
            axes[i].axis('off')

    # Turn off any unused subplots
    for j in range(num_images, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'batch_0.pdf'))
    plt.savefig(os.path.join(model_folder, 'batch_0.png'))

    if verbose :
        print(f"Batch 0 saved as '{os.path.join(model_folder, 'batch_0.png')}'")

    # Define the mean and standard deviation for ImageNet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Create a Sequential model with all layers defined inside the constructor
    '''
    model = keras.models.Sequential([
      tf.keras.Input(shape=(image_shape[1], image_shape[0], 3)),
      keras.layers.Normalization(mean=mean, variance=[s**2 for s in std]),
      InceptionV3(weights='imagenet', include_top=False),
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(11, activation='sigmoid')
    ])
    '''
    
    '''
    input_layer = tf.keras.Input(shape=(image_shape[1], image_shape[0], 3))
    x = keras.layers.Normalization(mean=mean, variance=[s**2 for s in std])(input_layer)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=x)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(11, activation='sigmoid')(x)
    model = keras.models.Model(inputs=input_layer, outputs=output)
    '''
    
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_shape[1], image_shape[0], 3))
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(11, activation='sigmoid')(x)
    model = keras.models.Model(inputs=model.input, outputs=output)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                metrics=[tf.keras.metrics.F1Score(average='macro', threshold=None, name='macro_f1_score', dtype=None),
                        tf.keras.metrics.F1Score(average='micro', threshold=None, name='micro_f1_score', dtype=None)])

    model_filepath = os.path.join(model_folder, 'multilabel-model.keras')
    callbacks = [
      keras.callbacks.EarlyStopping(patience=50, monitor='val_micro_f1_score'),
      keras.callbacks.ModelCheckpoint(filepath=model_filepath, save_best_only=True),
      keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    if USE_WANDB :
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api = user_secrets.get_secret("wandb_api")
        import wandb
        from wandb.integration.keras import WandbMetricsLogger, WandbCallback, WandbEvalCallback
        wandb.login(key=wandb_api)
        wandb.init('Physionet-Challenge')
        callbacks += [WandbMetricsLogger()]

    # Load tensorboard
    #%load_ext tensorboard
    #%tensorboard --logdir logs

    history = model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks, validation_data=valid_dataset)

    ## Visualize training & validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig('loss.pdf')
    plt.savefig('loss.png')

    ## Visualize training & validation metrics
    plt.figure(figsize=(12, 6))
    metrics = ['macro_f1_score', 'micro_f1_score']
    for metric in metrics:
        train_metric = history.history[metric]
        val_metric = history.history[f'val_{metric}']
        plt.plot(train_metric, label=metric)
        plt.plot(val_metric, label=f'val_{metric}')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training & Validation Metrics')
    plt.legend()
    plt.savefig('metrics.pdf')
    plt.savefig('metrics.png')


# Load your trained models. This function is *required*. You should edit this
# function to add your code, but do *not* change the arguments of this
# function. If you do not train one of the models, then you can return None for
# the model.

def load_models(model_folder, verbose):
    digitization_model = None
    classification_filepath = os.path.join(model_folder, 'multilabel-model.keras')
    if not os.path.exists(classification_filepath):
        wget.download('https://storage.googleapis.com/figures-gp/physionet/multilabel-model.keras', model_folder)

    classification_model = tf.keras.models.load_model(classification_filepath)
    #classification_model = keras.saving.load_model(classification_filepath, custom_objects=None, compile=True, safe_mode=True)

    return digitization_model, classification_model


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):

    # Run the digitization model; if you did not train this model, then you can set signal = None.
    signal = None

    # Run the classification model
    if verbose:
        print(f"Running classification model on {record}")

    header_path = f'{record}.hea'
    with open(header_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        image_path = ""
        labels = ""

        # Iterate over each line in the .hea file
        for line in lines:
            if 'Labels' in line:
                raise LabelsNotRemovedError(header_path)
            if 'png' in line:
                image_name = line.split(":")[1].strip()

    # Open the image:
    record_parent_folder=os.path.dirname(header_path)
    #image_files=get_image_files(record)
    image_path=os.path.join(record_parent_folder, image_name)
    #img = Image.open(image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_shape)
    # FIXME: repeated code---maybe factor out opening the image from a record
    #if img.mode != 'RGB':
    #    img = img.convert('RGB')

    images = np.array([img])

    #inference
    output = classification_model.predict(images)

    #get all labels
    class_array = (output >= 0.5).astype(int)

    # Find the indices where the value is 1
    indices = np.where(class_array[0] == 1)[0]

    # Map indices to class labels
    predictions = [LABELS[i] for i in indices]

    #TODO: backup if none is over the threshold: use the max
    if predictions==[]:
        ## do something to avoid returning an empty list
        ## Pick the most likely class or n most likely classes
        pass
    if verbose:
        print(f'Image Name: {image_name}')
        print(f'Predicted Labels: {predictions}\n\n')

    return signal, predictions



#########################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
#########################################################################################

##############################################################################################################
##################################  SUPPLEMENTARY FUNCTIONS  #################################################
def extract_info_from_hea(label_mapping, physio_paths):
    data = []

    # Iterate over all files in the given folder
    for filename in physio_paths:
        if filename.endswith(".hea"):
            file_path = filename #os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                image_path = ""
                labels = ""

                # Iterate over each line in the .hea file
                for line in lines:
                    if 'Labels' in line:
                        labels = line.split(":")[1].strip()
                    if 'png' in line:
                        image_path = line.split(":")[1].strip()

                # Create a dictionary to hold the image path and label information
                label_info = {label: 0 for label in label_mapping}

                if labels:
                    for label in labels.split(","):
                        if label.strip() in label_mapping:
                            label_info[label.strip()] = 1

                label_info["Image_Name"] = image_path

                #get the head and tail path
                head, tail = os.path.split(filename)
                new_path = os.path.join(head, image_path)
                label_info["Image_Path"] = new_path
                data.append(label_info)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    return df

################################################################################
##################### DATA LOADER FUNCTION #####################################

def get_dataset(
    batch_size,
    img_size,
    df,
    max_dataset_len=None,
):
    """Returns a TF Dataset."""

    input_img_paths = df['Image_Path'].values
    labels = df[LABELS].values

    def load_images(input_img_path, label):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, (img_size[1], img_size[0]))
        input_img = tf_image.convert_image_dtype(input_img, "float32")
        #label = tf.cast(label, tf.float32)
        return input_img, label

    # For faster debugging, limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        labels = labels[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, labels))
    dataset = dataset.map(load_images, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


class LabelsNotRemovedError(Exception):
    def __init__(self, header_path):
        super().__init__(f'''\n\nThe labels have not been removed from the header file {header_path}
                          Run the vanilla-cnn-2024/remove_hidden_data.py script to remove the labels''')
        self.header_path = header_path
