import tensorflow as tf
import numpy as np
from GTSRB_utils import GTSRB_CLASSES, predict_traffic_sign

def generate_uap(model, dataloader, epsilon=0.2, num_epochs=7):
    """
    Generate a Universal Adversarial Perturbation (UAP) for a given model and dataset (TensorFlow version).

    Parameters:
        model (tf.keras.Model): The target deep learning model.
        dataloader (tf.data.Dataset): Dataset for training images.
        epsilon (float): Maximum perturbation magnitude.
        num_epochs (int): Number of epochs for perturbation training.

    Returns:
        tf.Tensor: The generated universal perturbation.
    """
    # Get input shape from dataset
    for images, _ in dataloader.take(1):
        input_shape = images.shape[1:]

    # Initialize universal perturbation
    delta = tf.Variable(tf.zeros(input_shape), trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in dataloader:
            with tf.GradientTape() as tape:
                perturbed_images = tf.clip_by_value(images + delta, 0, 1)
                predictions = model(perturbed_images, training=False)
                loss = -loss_fn(labels, predictions)  # Maximize the loss

            gradients = tape.gradient(loss, delta)
            optimizer.apply_gradients([(gradients, delta)])

            # Ensure perturbation remains within bounds
            delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))
            epoch_loss += loss.numpy()

        print(f'UAP Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(dataloader):.4f}')

    return delta.numpy()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def apply_uap(model, images, delta, class_mapping):
    """
    Apply a trained universal perturbation to a batch of images and predict both original and adversarial labels.

    Parameters:
        model (tf.keras.Model): The target deep learning model.
        images (tf.Tensor or np.ndarray): Batch of input images.
        delta (tf.Tensor): Universal perturbation.
        class_mapping (dict): Mapping from class indices to class names.

    Returns:
        list: A list of dictionaries containing original and adversarial predictions.
    """
    # Ensure images and delta are in [0,1] range
    if images.max() > 1:
        images_new = images / 255.0
    if delta.max() > 1:
        delta = delta / 255.0

    # Apply perturbation and clip values to valid range
    perturbed_images = tf.clip_by_value(images_new + delta, 0, 1).numpy()  # Convert to NumPy for visualization

    results = []

    for i in range(len(images)):
        original_class, original_conf = predict_traffic_sign(images[i:i+1], model, class_mapping)
        adversarial_class, adversarial_conf = predict_traffic_sign(perturbed_images[i:i+1], model, class_mapping)

        result = {
            "image_index": i + 1,
            "original_prediction": original_class,
            "original_confidence": original_conf,
            "adversarial_prediction": adversarial_class,
            "adversarial_confidence": adversarial_conf
        }
        results.append(result)

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].imshow(images[i]/255)  # No need to divide by 255 if already in [0,1]
        ax[0].set_title(f"Original: {original_class}\nConf: {original_conf:.4f}")
        ax[0].axis("off")

        ax[1].imshow(perturbed_images[i])
        ax[1].set_title(f"Adversarial: {adversarial_class}\nConf: {adversarial_conf:.4f}")
        ax[1].axis("off")

        plt.show()  # Display the images properly

        print(f"Image {i+1}:")
        print(f"  Original Prediction: {original_class} (Confidence: {original_conf:.4f})")
        print(f"  Adversarial Prediction: {adversarial_class} (Confidence: {adversarial_conf:.4f})")
        print("-" * 50)

    return results

