import tensorflow as tf

def deepfool_attack(model, image, true_label, max_iter=50, overshoot=0.02):
    """
    DeepFool attack implementation for a single image classification (e.g., EfficientNet).
    
    Args:
        model: A Keras classification model that outputs logits.
        image: A tensor of shape [H, W, C] with values in [0, 1] (preprocessed as required).
        true_label: An integer representing the correct class index.
        max_iter: Maximum number of iterations.
        overshoot: Maximum allowed perturbation (per pixel) for the accumulated perturbation.
        
    Returns:
        pert_image: The perturbed (adversarial) image with shape [H, W, C].
        r_total: The total perturbation applied, also with shape [H, W, C].
    """
    # Convert inputs to tensors
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    true_label = tf.convert_to_tensor(true_label, dtype=tf.int32)
    
    # Add a batch dimension if the image is single (shape: [H, W, C] -> [1, H, W, C])
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
    if len(true_label.shape) == 0:
        true_label = tf.expand_dims(true_label, axis=0)
    
    # Initialize the total perturbation (r_total) as zeros
    r_total = tf.zeros_like(image)
    pert_image = tf.identity(image)
    
    # Small constant to prevent division by zero
    epsilon_val = 1e-4
    
    i = 0
    while i < max_iter:
        with tf.GradientTape() as tape:
            tape.watch(pert_image)
            logits = model(pert_image, training=False)
            # Compute loss using sparse categorical crossentropy (from logits)
            per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(true_label, logits, from_logits=False)
            loss = tf.reduce_mean(per_example_loss)
        
        # Compute gradients of loss with respect to pert_image
        grad = tape.gradient(loss, pert_image)
        # Normalize gradient per example
        grad_reshaped = tf.reshape(grad, [grad.shape[0], -1])
        norm = tf.norm(grad_reshaped, axis=1, keepdims=True) + 1e-8
        norm = tf.reshape(norm, [-1, 1, 1, 1])
        w = grad / norm
        
        # Expand per-example loss for broadcasting
        loss_expanded = tf.reshape(per_example_loss, [-1, 1, 1, 1])
        # Compute the perturbation step for each example
        r_i = (loss_expanded + epsilon_val) * w
        
        # Update the accumulated perturbation and clip to allowed range
        r_total = tf.clip_by_value(r_total + r_i, -overshoot, overshoot)
        # Update the perturbed image and clip pixel values to [0, 1]
        pert_image = tf.clip_by_value(image + r_total, 0.0, 1.0)
        
        # Compute predictions using argmax over logits
        preds = tf.argmax(model(pert_image, training=False), axis=1, output_type=tf.int32)
        # Stop if the predicted label is no longer equal to the true label
        if tf.reduce_all(tf.not_equal(preds, true_label)):
            break
        
        i += 1
        print(f"Iteration {i}: {preds.numpy()[0]}")

    
    
    return pert_image, r_total
