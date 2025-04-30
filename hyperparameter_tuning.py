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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import visualkeras
from PIL import Image
# Configuration flag to control execution mode
training_mode = False  # Set to True for training, False for evaluation
model_weights_path = '/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/hyperparameter_tuning_5_new_classes_1500_compressed/densenet121_tuning/trial_01'  # Path to the directory containing checkpoint.weight.h5

# Manual hyperparameter specification (set to None to auto-detect from trial directory)
# If you know the exact hyperparameters you want to use, set them here
MANUAL_HYPERPARAMS = {
    'l2_reg': None,         # Example: 1e-4
    'dropout_rate': None,   # Example: 0.5
    'dense_units_1': None,  # Example: 1024
    'dense_units_2': None,  # Example: 512
    'dense_units_3': None,  # Example: 256
    'learning_rate': None   # Example: 1e-4
}
# Set to True to use manual hyperparameters, False to auto-detect
USE_MANUAL_HYPERPARAMS = False

# -----------------------------
# A) Directory Setup & Hyperparams
# -----------------------------
OUTPUT_DIR = '/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/Balanced_5_classes'
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Nodule', 'Pneumothorax','Effusion']

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
test_datagen = ImageDataGenerator(rescale=1.0/255)

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

# Function to create the model architecture for loading weights
def create_base_model(l2_reg=1e-4, dropout_rate=0.5, dense_units_1=1024, dense_units_2=512, dense_units_3=256, learning_rate=1e-4):
    """
    Create a DenseNet121 model with customizable hyperparameters.
    
    Args:
        l2_reg: L2 regularization strength
        dropout_rate: Dropout rate for regularization
        dense_units_1: Number of units in first dense layer
        dense_units_2: Number of units in second dense layer
        dense_units_3: Number of units in third dense layer
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
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
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if training_mode:
    # -----------------------------
    # D) Tuner Setup
    # -----------------------------
    # Configuration flag - Set to True to start fresh, False to resume previous work
    START_FRESH = True  # Change this to False when you want to resume previous work

    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=10,  # Number of trials to run
        directory='hyperparameter_tuning',
        project_name='densenet121_tuning',
        overwrite=START_FRESH,  # Overwrite if starting fresh
        num_initial_points=3  # Number of random trials before starting Bayesian optimization
    )

    # Only reload if we're not starting fresh
    if not START_FRESH:
        tuner.reload()
        print(f"Resuming previous work. Trials completed so far: {len(tuner.oracle.trials)}")
    else:
        print("Starting fresh hyperparameter optimization")

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
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
        tensorboard_cb
    ]

    # -----------------------------
    # F) Run Hyperparameter Search
    # -----------------------------
    print(f"\nRunning hyperparameter search. TensorBoard logs will be saved to: {log_dir}")
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
        epochs=100  # Fixed epochs
    )
    
    print("Hyperparameter search completed. Set training_mode = False to evaluate the model.")

else:
    # -----------------------------
    # Load and Evaluate Pre-trained Model
    # -----------------------------
    print("\nLoading pre-trained model for evaluation...")
    
    # Set hyperparameters
    if USE_MANUAL_HYPERPARAMS and any(value is not None for value in MANUAL_HYPERPARAMS.values()):
        print("\nUsing manually specified hyperparameters:")
        l2_reg = MANUAL_HYPERPARAMS['l2_reg'] if MANUAL_HYPERPARAMS['l2_reg'] is not None else 1e-4
        dropout_rate = MANUAL_HYPERPARAMS['dropout_rate'] if MANUAL_HYPERPARAMS['dropout_rate'] is not None else 0.5
        dense_units_1 = MANUAL_HYPERPARAMS['dense_units_1'] if MANUAL_HYPERPARAMS['dense_units_1'] is not None else 1024
        dense_units_2 = MANUAL_HYPERPARAMS['dense_units_2'] if MANUAL_HYPERPARAMS['dense_units_2'] is not None else 512
        dense_units_3 = MANUAL_HYPERPARAMS['dense_units_3'] if MANUAL_HYPERPARAMS['dense_units_3'] is not None else 256
        learning_rate = MANUAL_HYPERPARAMS['learning_rate'] if MANUAL_HYPERPARAMS['learning_rate'] is not None else 1e-4
        
        print(f"L2 Regularization: {l2_reg}")
        print(f"Dropout Rate: {dropout_rate}")
        print(f"Dense Units 1: {dense_units_1}")
        print(f"Dense Units 2: {dense_units_2}")
        print(f"Dense Units 3: {dense_units_3}")
        print(f"Learning Rate: {learning_rate}")
    else:
        # Get best hyperparameters from trial directories
        try:
            # Try to load best hyperparameters from trial.json files
            import json
            import glob
            
            # First, check if the current path points directly to a trial directory with trial.json
            if os.path.exists(os.path.join(model_weights_path, 'trial.json')):
                # We're already in a trial directory
                trial_path = os.path.join(model_weights_path, 'trial.json')
                with open(trial_path, 'r') as f:
                    trial_data = json.load(f)
                
                # Extract hyperparameters
                hyperparameters = trial_data.get('hyperparameters', {}).get('values', {})
                l2_reg = hyperparameters.get('l2_reg', 1e-4)
                dropout_rate = hyperparameters.get('dropout_rate', 0.5)
                dense_units_1 = hyperparameters.get('dense_units_1', 1024)
                dense_units_2 = hyperparameters.get('dense_units_2', 512)
                dense_units_3 = hyperparameters.get('dense_units_3', 256)
                learning_rate = hyperparameters.get('learning_rate', 1e-4)
                score = trial_data.get('score', 0)
                
                print(f"\nLoaded hyperparameters from trial.json (score: {score}):")
                print(f"L2 Regularization: {l2_reg}")
                print(f"Dropout Rate: {dropout_rate}")
                print(f"Dense Units 1: {dense_units_1}")
                print(f"Dense Units 2: {dense_units_2}")
                print(f"Dense Units 3: {dense_units_3}")
                print(f"Learning Rate: {learning_rate}")
            else:
                # Look for trial directories in the parent directory
                parent_dir = os.path.dirname(model_weights_path)
                trial_dirs = glob.glob(os.path.join(parent_dir, 'trial_*'))
                
                if not trial_dirs:
                    # If no trial directories found, try to use the parent directory
                    trial_dirs = glob.glob(os.path.join(os.path.dirname(parent_dir), 'trial_*'))
                
                if trial_dirs:
                    print(f"\nFound {len(trial_dirs)} trial directories. Searching for best hyperparameters...")
                    
                    # Find the best trial based on score
                    best_score = -float('inf')
                    best_hyperparams = None
                    best_trial_dir = None
                    
                    for trial_dir in trial_dirs:
                        trial_path = os.path.join(trial_dir, 'trial.json')
                        
                        if os.path.exists(trial_path):
                            with open(trial_path, 'r') as f:
                                trial_data = json.load(f)
                            
                            score = trial_data.get('score', 0)
                            
                            if score > best_score:
                                best_score = score
                                best_hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
                                best_trial_dir = trial_dir
                    
                    if best_hyperparams:
                        # Extract hyperparameters from best trial
                        l2_reg = best_hyperparams.get('l2_reg', 1e-4)
                        dropout_rate = best_hyperparams.get('dropout_rate', 0.5)
                        dense_units_1 = best_hyperparams.get('dense_units_1', 1024)
                        dense_units_2 = best_hyperparams.get('dense_units_2', 512)
                        dense_units_3 = best_hyperparams.get('dense_units_3', 256)
                        learning_rate = best_hyperparams.get('learning_rate', 1e-4)
                        
                        print(f"\nFound best hyperparameters from {os.path.basename(best_trial_dir)} (score: {best_score}):")
                        print(f"L2 Regularization: {l2_reg}")
                        print(f"Dropout Rate: {dropout_rate}")
                        print(f"Dense Units 1: {dense_units_1}")
                        print(f"Dense Units 2: {dense_units_2}")
                        print(f"Dense Units 3: {dense_units_3}")
                        print(f"Learning Rate: {learning_rate}")
                    else:
                        print("\nCould not find trial.json in any trial directory. Using default parameters.")
                        l2_reg, dropout_rate = 1e-4, 0.5
                        dense_units_1, dense_units_2, dense_units_3 = 1024, 512, 256
                        learning_rate = 1e-4
                else:
                    print("\nNo trial directories found. Using default parameters.")
                    l2_reg, dropout_rate = 1e-4, 0.5
                    dense_units_1, dense_units_2, dense_units_3 = 1024, 512, 256
                    learning_rate = 1e-4
                    
        except Exception as e:
            print(f"\nError loading hyperparameters: {e}. Using default parameters.")
            l2_reg, dropout_rate = 1e-4, 0.5
            dense_units_1, dense_units_2, dense_units_3 = 1024, 512, 256
            learning_rate = 1e-4
    
    # Load model architecture with the best hyperparameters
    model = create_base_model(
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2, 
        dense_units_3=dense_units_3,
        learning_rate=learning_rate
    )
    
    # Load weights
    weights_path = os.path.join(model_weights_path, 'checkpoint.weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")
    
    model.load_weights(weights_path)
    print(f"Model weights loaded from {weights_path}")


    
    architecture_img = visualkeras.layered_view(model, legend=True, draw_volume=True, show_dimension=True)

    architecture_img.save('model_architecture.png')
    print("Model architecture visualization saved to model_architecture.png")
    
    # Generate test data
    print("\nEvaluating model on test set...")
    test_generator = test_datagen.flow_from_directory(
        directory=TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=8,
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
    plt.title("Confusion Matrix - Evaluation")
    plt.tight_layout()
    plt.savefig('confusion_matrix_evaluation.png')
    plt.close()
    
    # Generate classification report (precision, recall, f1-score)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    
    # Print classification report in a nicely formatted way
    print("\nClassification Report:")
    print("="*60)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-"*60)
    
    for class_name in class_labels:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    print("-"*60)
    # Print averages
    print(f"{'Accuracy':<15} {'':<10} {'':<10} {'':<10} {report['accuracy']:.4f}")
    print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<10.4f} {report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f} {report['macro avg']['support']:<10}")
    print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<10.4f} {report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f} {report['weighted avg']['support']:<10}")
    print("="*60)
    
    # Also save the classification report to a file
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    print("\nEvaluation complete. Results saved to:")
    print("- confusion_matrix_evaluation.png")
    print("- classification_report.txt") 