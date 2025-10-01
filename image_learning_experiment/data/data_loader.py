import pickle
import numpy as np
import os
import sys
from .. import config # Use relative import within the package

# Ensure the config module can be found by adding the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

def unpickle(file):
    """Load byte data from file."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(filename):
    """Load a single batch of cifar data."""
    datadict = unpickle(filename)
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y

def preprocess_image_to_receptive_fields(image, rf_size, rf_stride):
    """Convert a 32x32x3 image to a vector of receptive field activations."""
    # Convert to grayscale (average over color channels)
    # Input shape: (32, 32, 3)
    if image.ndim == 3 and image.shape[2] == 3:
        grayscale_image = image.mean(axis=2)
    elif image.ndim == 2:
        grayscale_image = image # Assume already grayscale
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Normalize grayscale image to [0, 1]
    min_val, max_val = np.min(grayscale_image), np.max(grayscale_image)
    if max_val > min_val:
        grayscale_image = (grayscale_image - min_val) / (max_val - min_val)
    else:
        grayscale_image = np.zeros_like(grayscale_image) # Handle case of uniform color image

    img_h, img_w = grayscale_image.shape
    rf_activations = []

    for y in range(0, img_h - rf_size + 1, rf_stride):
        for x in range(0, img_w - rf_size + 1, rf_stride):
            patch = grayscale_image[y:y+rf_size, x:x+rf_size]
            rf_activations.append(np.mean(patch))

    return np.array(rf_activations)

def load_cifar10_data(path):
    """Load all CIFAR-10 data and preprocess using receptive fields."""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, f'data_batch_{b}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(path, 'test_batch'))

    # Preprocess images using receptive fields
    Xtr_processed = np.array([preprocess_image_to_receptive_fields(img, config.RF_SIZE, config.RF_STRIDE) for img in Xtr])
    Xte_processed = np.array([preprocess_image_to_receptive_fields(img, config.RF_SIZE, config.RF_STRIDE) for img in Xte])

    # Ensure the output shape matches the configuration
    expected_size = config.PREPROCESS_TRAIN_TARGET_SIZE
    if Xtr_processed.shape[1] != expected_size:
        raise ValueError(f"Processed training data has unexpected feature size: {Xtr_processed.shape[1]}, expected {expected_size}")
    if Xte_processed.shape[1] != expected_size:
        raise ValueError(f"Processed test data has unexpected feature size: {Xte_processed.shape[1]}, expected {expected_size}")

    return Xtr_processed, Ytr, Xte_processed, Yte

# Deprecated: Original preprocessing function (flattening)
# def preprocess_images(images, target_size):
#     """Basic preprocessing: flatten and normalize."""
#     num_images = images.shape[0]
#     images_flat = images.reshape(num_images, -1).astype(np.float32)

#     # Normalize each image individually to range [0, 1]
#     min_vals = np.min(images_flat, axis=1, keepdims=True)
#     max_vals = np.max(images_flat, axis=1, keepdims=True)
#     range_vals = max_vals - min_vals
#     # Avoid division by zero for uniform images
#     range_vals[range_vals == 0] = 1
#     images_normalized = (images_flat - min_vals) / range_vals

#     if images_normalized.shape[1] != target_size:
#          raise ValueError(f"Flattened image size {images_normalized.shape[1]} does not match target size {target_size}")
#     return images_normalized

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Construct the path relative to this file's location
    cifar_dir = os.path.join(config.DATA_ROOT, 'cifar-10-batches-py')
    print(f"Looking for CIFAR data in: {os.path.abspath(cifar_dir)}")

    if not os.path.exists(cifar_dir):
        print("CIFAR-10 data directory not found. Please download and extract it.")
        print(f"Expected location: {os.path.abspath(cifar_dir)}")
    else:
        X_train, y_train, X_test, y_test = load_cifar10_data(cifar_dir)
        print('Training data shape: ', X_train.shape)
        print('Training labels shape: ', y_train.shape)
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', y_test.shape)

        # Example preprocessing
        # Assuming LOW_LEVEL area expects 3072 inputs (32*32*3)
        target_input_size = 3072 # Example, should match config.AREA_CONFIG[config.LOW_LEVEL]['n'] if using direct pixels
        processed_test_images = preprocess_image_to_receptive_fields(X_test, config.RF_SIZE, config.RF_STRIDE)
        print('Processed test data shape: ', processed_test_images.shape) 