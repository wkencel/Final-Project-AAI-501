"""
Title: [KerasCV] Image classification using a CNN with Keras/Tensorflow
Description: Training an image classifier from scratch on the Kaggle Trash Images dataset.
Accelerator: GPU
"""
"""
## Introduction

This example shows how to do image classification from scratch, starting from JPEG
image files on disk, without leveraging pre-trained weights or a pre-made Keras
Application model. We demonstrate the workflow on the Kaggle Trash Images binary
 classification dataset.

We use the `image_dataset_from_directory` utility to generate the datasets, and
we use Keras image preprocessing layers for image standardization and data augmentation.
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_cv
import matplotlib.pyplot as plt

"""

## Load the data: the Trash Images dataset
## Raw data located in data/trash_images

## Filter out corrupted images

When working with lots of real-world image data, corrupted images are a common
occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
in their header.
"""

import os

num_skipped = 0
for folder_name in ("cardboard", "glass", "metal", "paper", "plastic", "trash"):
    folder_path = os.path.join("..", "data", "trash_images", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

"""
## Generate a `Dataset`
"""

image_size = (180, 180)
batch_size = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/trash_images",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

"""
## Visualize the data

Here are the first 8 images in the training dataset, visualized using
the KerasCV plot_image_gallery utility.
"""

vis_ds = train_ds.take(1).unbatch()

vis_ds = vis_ds.take(8)


def get_images(image, _):
    return image


vis_ds = vis_ds.map(get_images)

vis_ds = vis_ds.ragged_batch(batch_size=8)

keras_cv.visualization.plot_image_gallery(
    next(iter(vis_ds.take(1))),
    value_range=(0, 255),
    scale=3,
    rows=4,
    cols=2,
)

"""
## Using image data augmentation

When you don't have a large image dataset, it's a good practice to artificially
introduce sample diversity by applying random yet realistic transformations to the
training images, such as random horizontal flipping or small random rotations. This
helps expose the model to different aspects of the training data while slowing down
overfitting. For this, we can make use of KerasCV and its wide array of preprocessing
layers.
"""

data_augmentation = keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(
            value_range=(0, 255),
            augmentations_per_image=2,
            magnitude=0.5,
            magnitude_stddev=0.15,
        ),
    ]
)

"""
Let's visualize what the augmented samples look like, by applying `data_augmentation`
repeatedly to the first few images in the dataset:
"""

vis_ds = vis_ds.map(data_augmentation)

keras_cv.visualization.plot_image_gallery(
    next(iter(vis_ds.take(1))),
    value_range=(0, 255),
    scale=3,
    rows=4,
    cols=2,
)

"""
## Standardizing the data

Our image are already in a standard size (180x180), as they are being yielded as
contiguous `float32` batches by our dataset. However, their RGB channel values are in
the `[0, 255]` range. This is not ideal for a neural network;
in general you should seek to make your input values small. Here, we will
standardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of
our model.
"""

"""
## Preprocess the data

Use the `data_augmentation` preprocessor:

Your data augmentation will happen **on CPU**, asynchronously, and will
be buffered before going into the model.

If you're training on CPU, this is the better option, since it makes data augmentation
asynchronous and non-blocking.

In our case, we'll go with the second option. If you're not sure
which one to pick, this second option (asynchronous preprocessing) is always a solid choice.
"""

"""
## Configure the dataset for performance

Let's apply data augmentation to our training dataset,
and let's make sure to use buffered prefetching so we can yield data from disk without
having I/O becoming blocking:
"""
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

"""
## Build a model

build a small version of the Xception network. We haven't particularly tried to
optimize the architecture; if you want to do a systematic search for the best model
configuration, consider using
[KerasTuner](https://github.com/keras-team/keras-tuner).

** This configuration has a Bayesian OPtimizer that may be worth looking at to see if it's useful for our model **

Note that:

- We start the model with the `data_augmentation` preprocessor, followed by a
 `Rescaling` layer.
- We include a `Dropout` layer before the final classification layer.
"""


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

"""
## Train the model
"""

# for testing, only use 5 epochs to save time
epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.legacy.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

"""
We get to >90% validation accuracy after training for 25 epochs on the full dataset
(in practice, you can train for 50+ epochs before validation performance starts degrading).
"""

"""
## Run inference on new data

Note that data augmentation and dropout are inactive at inference time.
"""

img = keras.utils.load_img("../data/trash_images/cardboard/cardboard_015.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])

# print the output

# Just for illustration, replace 0.5 with your score value
score = 0.5
# Define the materials as list
materials = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Generating the message
output_strings = [f"{100 * (1 - score):.2f}% {material}" if material == "cardboard" 
                  else f"{100 * score:.2f}% {material}" for material in materials]

# Join and print
output = " and ".join(output_strings)
print(f"This image is {output}.")
