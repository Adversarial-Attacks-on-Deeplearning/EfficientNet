import tensorflow as tf
import numpy as np
from GTSRB_utils import GTSRB_CLASSES, predict_traffic_sign

import tensorflow as tf

def generate_uap(model, dataloader, epsilon=5.0, num_epochs=7):
    """
    Generate a Universal Adversarial Perturbation (UAP) for a given model and dataset,
    for models that expect unnormalized images (pixel values in [0, 255]).

    Parameters:
        model (tf.keras.Model): The target deep learning model.
        dataloader (tf.data.Dataset): Dataset yielding (images, labels), where images are in [0,255].
        epsilon (float): Maximum perturbation magnitude (in pixel-value units, e.g. 10 means +/-10 intensity).
        num_epochs (int): Number of epochs for perturbation training.

    Returns:
        tf.Tensor: The generated universal perturbation (same shape as one input image).
    """
    # Infer input shape from one batch
    for images, _ in dataloader.take(1):
        input_shape = images.shape[1:]
        # Ensure float32 for perturbation math
        images = tf.cast(images, tf.float32)
    # Initialize universal perturbation
    delta = tf.Variable(tf.zeros(input_shape, dtype=tf.float32), trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0)  # you may tune LR
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels in dataloader:
            # Cast images to float32 for addition
            images = tf.cast(images, tf.float32)
            with tf.GradientTape() as tape:
                # Add the universal delta and clip to valid pixel range [0, 255]
                perturbed = tf.clip_by_value(images + delta, 0.0, 255.0)
                # Forward pass
                logits = model(perturbed, training=False)
                # We maximize loss, so negate the normal loss
                loss = -loss_fn(labels, logits)

            # Compute gradients w.r.t. delta
            grads = tape.gradient(loss, delta)
            optimizer.apply_gradients([(grads, delta)])

            # Project delta back into the L-infinity ball of radius epsilon
            delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))

            epoch_loss += loss.numpy()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f'UAP Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}')
        print(f'Delta range: min={tf.reduce_min(delta).numpy():.4f}, max={tf.reduce_max(delta).numpy():.4f}')

    # Return as a numpy array (you can cast to uint8 if you need an integer mask)
    return delta.numpy()

