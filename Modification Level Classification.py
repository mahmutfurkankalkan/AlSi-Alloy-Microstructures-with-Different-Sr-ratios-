# Import necessary libraries
import tensorflow as tf



from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

# Set up constants for the model
img_width, img_height = 180, 180
train_data_dir = '/content/drive/MyDrive/Train/Train'
validation_data_dir = '/content/drive/MyDrive/Validation/Validation'
nb_train_samples = 383
nb_validation_samples = 82
epochs = 50
batch_size = 16

# Check if the backend is Theano or TensorFlow
if K.image_data_format() == 'channels_first':
   input_shape = (3, img_width, img_height)
else:
   input_shape = (img_width, img_height, 3)

   # Define the model
   from keras.applications.densenet import DenseNet121

   model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

   # Freeze the layers
   for layer in model.layers:
       layer.trainable = False

   # Add new layers
   x = model.output
   x = Flatten()(x)
   x = Dense(1024, activation='relu')(x)
   x = Dropout(0.2)(x)
   predictions = Dense(5, activation='softmax')(x)

   # Create new model
   model = Model(inputs=model.input, outputs=predictions)


optimizer = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up image data generators for the training and validation sets
train_datagen = ImageDataGenerator(
rescale=1. / 255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
vertical_flip=True,
fill_mode='nearest',
rotation_range=40,  # Döndürme
brightness_range=[0.5,1.5]
)

test_datagen = ImageDataGenerator(
rescale=1. / 255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
vertical_flip=True,
fill_mode='nearest',
rotation_range=40,  # Döndürme
brightness_range=[0.5,1.5])
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# Save the model weights with the correct extension
model.save_weights('modlevel.weights.h5')
model.save("modleveldensenet201.h5")

# Create an image generator for the evaluation set
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory(
    '/content/drive/MyDrive/Test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42)

# Evaluate the model on the evaluation set
eval_loss, eval_acc = model.evaluate(eval_generator, steps=19)
print('Evaluation loss:', eval_loss)
print('Evaluation accuracy:', eval_acc)

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Extract evaluation results by class
predictions = model.predict(eval_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = eval_generator.classes
class_labels = list(eval_generator.class_indices.keys())

# Create a confusion matrix
confusion_matrix = sk_confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix as a heatmap
plt.imshow(confusion_matrix, cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.colorbar()
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Calculate the confusion matrix
confusion_matrix_result = confusion_matrix(true_classes, predicted_classes)

# Create and display the confusion matrix
ConfusionMatrixDisplay(confusion_matrix_result, display_labels=class_labels).plot()