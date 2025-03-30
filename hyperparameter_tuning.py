import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import keras_tuner as kt
import datetime

# -----------------------------
# A) Directory Setup & Hyperparams
# -----------------------------
OUTPUT_DIR = './balanced_dataset'
CLASSES = ['Atelectasis', 'Cardiomegaly', 'No Finding', 'Nodule', 'Pneumothorax']

TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR   = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR  = os.path.join(OUTPUT_DIR, 'test')

IMG_SIZE  = (512, 512)
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

# -----------------------------
# C) Model Building Function for Keras Tuner
# -----------------------------
def build_model(hp):
    # Hyperparameters to tune
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.7, step=0.1)
    dense_units_1 = hp.Int('dense_units_1', min_value=512, max_value=2048, step=256)
    dense_units_2 = hp.Int('dense_units_2', min_value=256, max_value=1024, step=256)
    dense_units_3 = hp.Int('dense_units_3', min_value=128, max_value=512, step=128)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    batch_size = hp.Choice('batch_size', values=[4, 8, 16, 32])

    # Build the model
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units_3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# -----------------------------
# D) Tuner Setup
# -----------------------------
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    directory='hyperparameter_tuning',
    project_name='densenet121_tuning'
)

# -----------------------------
# E) Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
    TensorBoard(log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
]

# -----------------------------
# F) Run Hyperparameter Search
# -----------------------------
tuner.search(
    train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=32,  # Initial batch size for search
        class_mode='categorical'
    ),
    validation_data=val_datagen.flow_from_directory(
        directory=VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    ),
    callbacks=callbacks,
    epochs=50
)

# -----------------------------
# G) Get Best Hyperparameters
# -----------------------------
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print(f"L2 Regularization: {best_hps.get('l2_reg')}")
print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Dense Units 1: {best_hps.get('dense_units_1')}")
print(f"Dense Units 2: {best_hps.get('dense_units_2')}")
print(f"Dense Units 3: {best_hps.get('dense_units_3')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print(f"Batch Size: {best_hps.get('batch_size')}")

# -----------------------------
# H) Train Final Model with Best Hyperparameters
# -----------------------------
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=best_hps.get('batch_size'),
        class_mode='categorical'
    ),
    validation_data=val_datagen.flow_from_directory(
        directory=VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=best_hps.get('batch_size'),
        class_mode='categorical'
    ),
    epochs=100,
    callbacks=callbacks
)

# Save the best model
model.save('best_model_tuned.h5') 