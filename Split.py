import os
import shutil

# Set the path of the original dataset directory
original_dataset_path = r'C:\Users\vaish\demo\Dataset_BUSI_with_GT'

# Set the path of the new directory for training and testing data
new_dataset_path = r'C:\Users\vaish\demo\model'
train_path = os.path.join(new_dataset_path, 'train')
test_path = os.path.join(new_dataset_path, 'test')

# Create the new directory structure
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get the list of class directories in the original dataset
class_directories = [d for d in os.listdir(original_dataset_path) if
                     os.path.isdir(os.path.join(original_dataset_path, d))]

# Split the data into training and testing sets
split_ratio = 0.8  # Set the ratio for splitting the data (80% training, 20% testing)

for class_directory in class_directories:
    class_path = os.path.join(original_dataset_path, class_directory)
    files = os.listdir(class_path)
    num_files = len(files)
    num_train = int(split_ratio * num_files)
    train_files = files[:num_train]
    test_files = files[num_train:]

    # Move the training files to the training directory
    for train_file in train_files:
        src = os.path.join(class_path, train_file)
        dst = os.path.join(train_path, class_directory, train_file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # Move the testing files to the testing directory
    for test_file in test_files:
        src = os.path.join(class_path, test_file)
        dst = os.path.join(test_path, class_directory, test_file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
