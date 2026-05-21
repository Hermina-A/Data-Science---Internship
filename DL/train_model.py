import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- 1. Hyperparameters & Configuration ---
BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = (150, 150)
DATASET_DIR = "Brain Tumor Data Set"

print("🔄 Initializing NeuroScan AI Core Data Pipeline...")

# --- 2. Image Augmentation & Split Configuration ---
# Setting validation_split allocates 20% of your data dynamically
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # Dynamic memory partition split
)

# --- 3. Loading Training & Validation Batches ---
print("\n📁 Extracting Training Partitions...")
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",  # Assigns the remaining 80% to train operations
    shuffle=True
)

print("\n📁 Extracting Validation Partitions...")
val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",  # Secures the isolated 20% validation pool
    shuffle=False
)

# --- 4. Build Convolutional Neural Network (CNN) ---
print("\n🛠️ Constructing Deep Learning Layers...")
model = Sequential([
    # Input Layer Group
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    # Hidden Layer Group 1
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    # Hidden Layer Group 2
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    # Processing Matrix Layer Group
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),  # Prevents overfitting metrics
    
    # Output Sigmoid node for automated classification indexing
    Dense(1, activation="sigmoid")
])

# --- 5. Compile and Optimize ---
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 6. Model Training Session ---
print(f"\n🚀 Launching training matrix over {EPOCHS} Epoch steps...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- 7. Export Weights Architecture Artifact ---
MODEL_NAME = "brain_tumor_model.h5"
print("\n💾 Finalizing structural architecture...")
model.save(MODEL_NAME)
print(f"✨ Success! Core binary engine saved safely as -> '{MODEL_NAME}'")