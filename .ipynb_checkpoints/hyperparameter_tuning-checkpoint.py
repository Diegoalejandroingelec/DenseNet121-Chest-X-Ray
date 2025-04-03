import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import keras_tuner as kt
import datetime
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
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,  # Number of trials to run
    directory='hyperparameter_tuning',
    project_name='densenet121_tuning'
)

# -----------------------------
# E) Callbacks
# -----------------------------
# Create a timestamp for unique log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('logs', timestamp)

# Enhanced TensorBoard callback with more metrics
tensorboard_cb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='batch',
    write_graph=True,
    write_images=True,
    profile_batch=2
)

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Callbacks for hyperparameter search (without checkpoint)
search_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
    tensorboard_cb
]

# Callbacks for final training (with checkpoint)
final_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
    tensorboard_cb,
    ModelCheckpoint(
        filepath=os.path.join('checkpoints', f'best_model_{timestamp}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# -----------------------------
# F) Run Hyperparameter Search
# -----------------------------
print(f"\nStarting hyperparameter search. TensorBoard logs will be saved to: {log_dir}")
print(f"To view TensorBoard, run: tensorboard --logdir {log_dir}")
tuner.search(
    train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=8,  # Fixed batch size
        class_mode='categorical'
    ),
    validation_data=val_datagen.flow_from_directory(
        directory=VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=8,  # Fixed batch size
        class_mode='categorical'
    ),
    callbacks=search_callbacks,  # Use search callbacks without checkpoint
    epochs=50  # Fixed epochs
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

# -----------------------------
# H) Train Final Model with Best Hyperparameters
# -----------------------------
print("\nTraining final model with best hyperparameters...")
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=8,  # Fixed batch size
        class_mode='categorical'
    ),
    validation_data=val_datagen.flow_from_directory(
        directory=VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=8,  # Fixed batch size
        class_mode='categorical'
    ),
    epochs=50,  # Fixed epochs
    callbacks=final_callbacks  # Use final callbacks with checkpoint
)

# Save the best model
model.save('best_model_tuned.h5')
print(f"\nTraining complete! Model saved as 'best_model_tuned.h5'")
print(f"To view training metrics in TensorBoard, run: tensorboard --logdir logs")

# -----------------------------
# I) Plot Training History
# -----------------------------
# Plot training history
plt.figure(figsize=(16, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# -----------------------------
# J) Evaluate on Test Set and Plot Confusion Matrix
# -----------------------------
print("\nEvaluating model on test set...")
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=8,  # Fixed batch size
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Get predictions
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Compute and plot confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Best Model")
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png')
plt.close()

# Print classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels)) 