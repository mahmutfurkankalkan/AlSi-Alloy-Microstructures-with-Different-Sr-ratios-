import os
from PIL import Image
import numpy as np

data_dir = "/content/drive/MyDrive/5x"

train_dir = os.path.join(data_dir, "Training")
val_dir = os.path.join(data_dir, "Validation")

train_images = []
train_masks = []

image_files = sorted(os.listdir(os.path.join(train_dir, "Image")))
mask_files = sorted(os.listdir(os.path.join(train_dir, "Mask")))

for filename in image_files:
    img = Image.open(os.path.join(train_dir, "Image", filename))

    if img.mode in ["RGB", "RGBA"]:
        img = img.convert("L")  # L mode, grayscale
        img.save(os.path.join(train_dir, "Image", filename.replace(".png", ".jpg")))

    train_images.append(np.array(img))

for filename in mask_files:
    mask = Image.open(os.path.join(train_dir, "Mask", filename))

    if mask is not None:
        mask = mask.convert("L")  # L mode, grayscale
        mask.save(os.path.join(train_dir, "Mask", filename.replace(".png", ".jpg")))
        train_masks.append(np.array(mask))
    else:
        print(f"Warning: Mask {filename} is None. Skipping...")

val_images = []
val_masks = []

image_files = sorted(os.listdir(os.path.join(val_dir, "Image")))
mask_files = sorted(os.listdir(os.path.join(val_dir, "Mask")))

for filename in image_files:
    img = Image.open(os.path.join(val_dir, "Image", filename))

    if img.mode in ["RGB", "RGBA"]:
        img = img.convert("L")  # L mode, grayscale
        img.save(os.path.join(val_dir, "Image", filename.replace(".png", ".jpg")))

    val_images.append(np.array(img))

for filename in mask_files:
    mask = Image.open(os.path.join(val_dir, "Mask", filename))

    if mask is not None:
        if mask.mode in ["RGB", "RGBA"]:
            mask = mask.convert("L")  # L mode, grayscale
            mask.save(os.path.join(val_dir, "Mask", filename.replace(".png", ".jpg")))
            val_masks.append(np.array(mask))
        else:
            val_masks.append(np.array(mask))
    else:
        print(f"Warning: Mask {filename} is None. Skipping...")


train_images = np.array(train_images)
train_masks = np.array(train_masks)
val_images = np.array(val_images)
val_masks = np.array(val_masks)

train_images_scaled = train_images / 255.0
train_masks_scaled = train_masks / 255.0
val_images_scaled = val_images / 255.0
val_masks_scaled = val_masks / 255.0


print("Image size:", train_images[0].shape)
print("Image size:", val_images[0].shape)
print("Image size:", train_masks[0].shape)
print("Image size:", val_masks[0].shape)
print("Image data type:", train_images[0].dtype)
print("Number of training images:", len(train_images))
print("Number of training masks:", len(train_masks))


import albumentations as A

transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
])

augmentation_factor = 8

aug_train_images = []
aug_train_masks = []
aug_val_images = []
aug_val_masks = []


for i in range(len(train_images)):
    if i >= len(train_masks):
        break  # Stop augmentation if train_images and train_masks have different lengths

    img = train_images[i]
    mask = train_masks[i]

    for j in range(augmentation_factor):
        augmented = transform(image=img, mask=mask)
        img_augmented = augmented['image']
        mask_augmented = augmented['mask']

        aug_train_images.append(img_augmented)
        aug_train_masks.append(mask_augmented)

for i in range(len(val_images)):
    img = val_images[i]
    mask = val_masks[i]

    for j in range(augmentation_factor):
        augmented = transform(image=img, mask=mask)
        img_augmented = augmented['image']
        mask_augmented = augmented['mask']

        aug_val_images.append(img_augmented)
        aug_val_masks.append(mask_augmented)

aug_train_images = np.array(aug_train_images)
aug_train_masks = np.array(aug_train_masks)
aug_val_images = np.array(aug_val_images)
aug_val_masks = np.array(aug_val_masks)

aug_train_images_scaled = aug_train_images / 255.0
aug_train_masks_scaled = aug_train_masks / 255.0
aug_val_images_scaled = aug_val_images / 255.0
aug_val_masks_scaled = aug_val_masks / 255.0

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def fcn(input_shape=(512, 512, 1), num_classes=1):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    up1 = Conv2D(256, 3, activation='relu', padding='same')(up1)
    up1 = Conv2D(256, 3, activation='relu', padding='same')(up1)

    up2 = concatenate([UpSampling2D(size=(2, 2))(up1), conv2], axis=-1)
    up2 = Conv2D(128, 3, activation='relu', padding='same')(up2)
    up2 = Conv2D(128, 3, activation='relu', padding='same')(up2)

    up3 = concatenate([UpSampling2D(size=(2, 2))(up2), conv1], axis=-1)
    up3 = Conv2D(64, 3, activation='relu', padding='same')(up3)
    up3 = Conv2D(64, 3, activation='relu', padding='same')(up3)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(up3)

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = fcn()


aug_train_images_expanded = np.expand_dims(aug_train_images_scaled, axis=-1)
aug_train_masks_expanded = np.expand_dims(aug_train_masks_scaled, axis=-1)
aug_val_images_expanded = np.expand_dims(aug_val_images_scaled, axis=-1)
aug_val_masks_expanded = np.expand_dims(aug_val_masks_scaled, axis=-1)

# For 5x (256x256), for 50x (512x512)
input_shape = (256, 256, 1)

model = fcn(input_shape)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(aug_train_images_expanded, aug_train_masks_expanded, validation_data=(aug_val_images_expanded, aug_val_masks_expanded), batch_size=8, epochs=200)


model_path = '/content/drive/MyDrive/segmentationfor5x.h5'
model.save(model_path)



