import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the previous layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=3, activation='softmax'))  # 3 classes: malignant, benign, normal

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values to [0,1]
    shear_range=0.2,      # Shear the image
    zoom_range=0.2,       # Zoom the image
    horizontal_flip=True  # Flip the image horizontally
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set the dataset directory and batch size
train_dir = r'C:\Users\vaish\demo\model\train'
test_dir = r'C:\Users\vaish\demo\model\test'
batch_size = 32

# Load the dataset
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Resize the images to (64, 64)
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical mode for multi-class classification
)

test_dataset = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_dataset,
    steps_per_epoch=train_dataset.samples // batch_size,
    epochs=20,  # Increase the number of epochs for better training
    validation_data=test_dataset,
    validation_steps=test_dataset.samples // batch_size


)
# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print('Test Loss:', loss*100)
print('Test Accuracy:', accuracy*100)


# Save the model
model.save('breast_cancer_model.h5')

