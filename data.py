import tensorflow as tf
import tensorflow_datasets as tfds

# Define the dataset name
dataset_name = 'cardiotox'  # Use the specific name of the dataset if 'cardiotox' is available in TFDS

# Check available versions and download the dataset
try:
    # Load the dataset
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True, download=True)
    print("Dataset loaded successfully")

    # Inspect the data splits and structure
    print("Dataset Info:", info)

    # Save dataset locally if required
    tfds_folder_path = tfds.core.utils.gcs_path('~/tensorflow_datasets')
    print("Downloaded dataset is saved at:", tfds_folder_path)

except tfds.core.DatasetNotFoundError:
    print(f"The dataset '{dataset_name}' is not available in TensorFlow Datasets. Verify the dataset name.")
