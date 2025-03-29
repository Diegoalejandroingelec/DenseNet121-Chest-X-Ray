import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# A) Directory Setup & Hyperparams
# -----------------------------
OUTPUT_DIR = './balanced_dataset'
CLASSES = ['Atelectasis', 'Cardiomegaly', 'No Finding', 'Nodule', 'Pneumothorax']

TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR   = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR  = os.path.join(OUTPUT_DIR, 'test')

IMG_SIZE  = (512, 512)
BATCH_SIZE = 8
NUM_CLASSES = len(CLASSES)

# -----------------------------
# B) Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
    channel_shift_range=0.1,
    fill_mode='reflect',
    preprocessing_function=lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2)
)

val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    directory=VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

# -----------------------------
# C) Load or Build Model
# -----------------------------
checkpoint_path = "best_model_modified.h5"
model = None

if os.path.exists(checkpoint_path):
    print(f"\n[INFO] Found existing checkpoint '{checkpoint_path}'. Loading model...")
    model = tf.keras.models.load_model(checkpoint_path)
    print("[INFO] Model loaded. Resuming training...\n")
else:
    print("[INFO] No existing checkpoint found. Building a new model...\n")
    
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = True  # Keep base model trainable

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

# Compile with higher learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

# -----------------------------
# D) Define Callbacks
# -----------------------------
log_dir = os.path.join("logs_1", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

checkpoint_cb = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks_list = [checkpoint_cb, reduce_lr_cb, earlystop_cb, tensorboard_cb]

# -----------------------------
# E) Train (or Continue Training)
# -----------------------------
EPOCHS = 100
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list
)

# -----------------------------
# F) Evaluate on Test Set
# -----------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Predict probabilities for the test set
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# Get the true class labels from the test generator
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Compute the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

