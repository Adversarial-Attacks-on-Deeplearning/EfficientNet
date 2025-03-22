import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path, target_size):
    """
    Reads an image from a file and resizes it to the target size.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired size (height, width) of the image.
    
    Returns:
        tf.Tensor: Resized image with pixel values in [0, 255].
    """
    # Read the image file
    image = tf.io.read_file(image_path)
    # Decode the image
    image = tf.image.decode_image(image, channels=3)
    # Resize the image to target size
    image = tf.image.resize(image, target_size)
    return tf.cast(image, tf.float32)  # Keep pixel values in [0, 255]


def save_image(image, image_path):
    """
    Saves an image to a file.
    
    Args:
        image (tf.Tensor): Image to save with pixel values in [0, 255].
        image_path (str): Path to save the image file.
    
    Returns:
        None
    """
    # Ensure pixel values are in [0, 255]
    image = tf.clip_by_value(image, 0, 255)
    # Convert to uint8
    image = tf.cast(image, tf.uint8)
    # Encode the image
    image = tf.image.encode_png(image)
    # Write the image to file
    tf.io.write_file(image_path, image)






def predict_and_plot(model, image, true_label):
    """
    Predicts the class of a given image using the model, and plots the image 
    with the predicted and true labels.
    
    Args:
        model (tf.keras.Model): Trained model.
        image (tf.Tensor): Input image of shape (H, W, C), values in [0, 255].
        true_label (int): Ground truth label for the image.
    
    Returns:
        None
    """
    # CIFAR-10 class names
    CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
    ]
    # Preprocess the image if necessary
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model(image, training=False).numpy()
    probabilities = predictions[0]
    
    # Get the predicted class
    predicted_class = tf.argmax(probabilities).numpy()
    predicted_label = CIFAR10_CLASSES[predicted_class]

    # Get the ground truth label
    true_label_name = CIFAR10_CLASSES[true_label]

    # Sort the probabilities to find the top 1%
    num_classes = len(probabilities)
    top_k = max(1, int(0.01 * num_classes))  # Ensure at least 1 class is displayed
    top_indices = np.argsort(probabilities)[-top_k:][::-1]  # Top 1% classes
    top_predictions = [(CIFAR10_CLASSES[i], probabilities[i]) for i in top_indices]

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image[0].numpy().astype("uint8"))
    plt.title(f"Predicted: {predicted_label}\nTrue: {true_label_name}")
    plt.axis('off')
    plt.show()

    # Display the top 1% predictions
    print("Top 1% Predictions:")
    for class_name, prob in top_predictions:
        print(f"  {class_name}: {prob * 100:.2f}%")





def is_adversarial(model, image, true_label):
    """
    Determines whether an image is adversarial by checking if the predicted class 
    is different from the true class.
    
    Args:
        model (tf.keras.Model): Trained model.
        image (tf.Tensor): Input image of shape (H, W, C), values in [0, 255].
        true_label (int): Ground truth label for the image.
    
    Returns:
        bool: True if the image is adversarial, False otherwise.
    """
    # Preprocess the image if necessary
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model(image, training=False).numpy()
    predicted_class = tf.argmax(predictions[0]).numpy()

    return predicted_class != true_label


