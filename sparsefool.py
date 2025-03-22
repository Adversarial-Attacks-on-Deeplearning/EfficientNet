from deep_fool import deepfool_attack
import tensorflow as tf
import numpy as np

def gradient_of_classifier(model, x, class_index=None):
    """
    Computes the gradient of the model's output with respect to x.
    
    Args:
        model: A Keras model.
        x: Input tensor with shape [batch_size, H, W, C]. (For a single image, batch_size=1.)
        class_index: (Optional) an integer or tensor specifying the class for which to compute the gradient.
                     If None, uses the predicted class.
                     
    Returns:
        grad: A tensor of the same shape as x representing the gradient.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=False)  # shape: [batch_size, num_classes]
        if class_index is None:
            class_index = tf.argmax(logits, axis=1, output_type=tf.int32)
        # Gather the logits for the selected class for each example.
        selected_logits = tf.gather_nd(logits,
            indices=tf.stack([tf.range(tf.shape(logits)[0]), class_index], axis=1))
    grad = tape.gradient(selected_logits, x)
    return grad

def sparse_fool_attack(model, image, true_label, 
                       deepfool_max_iter=50, sparse_max_iter=100, overshoot=0.02):
    """
    SparseFool-style attack for a single image classification.
    
    This function first obtains an initial adversarial perturbation using DeepFool,
    then refines it by iteratively modifying individual coordinates (pixels) to produce
    a sparse adversarial example.
    
    Args:
        model: A Keras classification model that outputs logits.
        image: A tensor of shape [H, W, C] with pixel values in [0,1].
        true_label: A scalar (or 0-D tensor) representing the correct class index.
        deepfool_max_iter: Maximum iterations for the DeepFool step.
        sparse_max_iter: Maximum iterations for the sparse refinement.
        overshoot: Maximum allowed perturbation (per pixel) for the initial DeepFool step.
        
    Returns:
        adv_image: The final adversarial image of shape [H, W, C].
        r_total: The sparse perturbation applied.
    """
    # Ensure inputs are tensors.
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    true_label = tf.convert_to_tensor(true_label, dtype=tf.int32)
    
    # Step 1: Get an initial adversarial perturbation using DeepFool.
    # Assume deepfool_attack_single returns (adv_image, r_adv)
    _, r_adv = deepfool_attack(model, image, true_label, max_iter=deepfool_max_iter, overshoot=overshoot)
    # Compute x_B, the approximate boundary point.
    x_B = image + r_adv

    # Step 2: Estimate the decision boundary normal at x_B.
    # We expand dims so that the helper works on a batched image.
    w = gradient_of_classifier(model, x_B)
    w = tf.squeeze(w, axis=0)  # Now shape [H, W, C]
    
    # Flatten image tensors for coordinate-wise operations.
    x_orig_flat = tf.reshape(image, [-1])
    x_current_flat = tf.reshape(image, [-1])
    x_B_flat = tf.reshape(x_B, [-1])
    w_flat = tf.reshape(w, [-1])
    
    # Initialize accumulated sparse perturbation.
    r_total_flat = tf.zeros_like(x_orig_flat)
    
    # Set to store indices that are saturated (i.e. already at 0 or 1).
    S = set()
    
    sparse_iter = 0
    # Continue until the model misclassifies x_current.
    # We wrap x_current in a batch dimension for prediction.
    while sparse_iter < sparse_max_iter:
        # Reshape current image to [1, H, W, C] for model prediction.
        x_current = tf.reshape(x_current_flat, image.shape)
        preds = tf.argmax(model(x_current, training=False), axis=1, output_type=tf.int32)
        # If the prediction has changed from the true label, break.
        if int(preds.numpy()[0]) != int(true_label.numpy()):
            break
        
        # Convert w_flat to a NumPy array to choose a coordinate index.
        w_flat_np = w_flat.numpy()
        valid_indices = [i for i in range(len(w_flat_np)) if i not in S]
        if not valid_indices:
            # All coordinates are saturated; exit loop.
            break
        
        # Select the coordinate index d (from valid_indices) with the maximum absolute w.
        abs_vals = [abs(w_flat_np[i]) for i in valid_indices]
        d = valid_indices[np.argmax(abs_vals)]
        
        # Compute the dot product between w_flat and (x_current_flat - x_B_flat)
        dot_val = tf.reduce_sum(w_flat * (x_current_flat - x_B_flat))
        numerator = tf.abs(dot_val)
        denominator = tf.abs(w_flat[d]) + 1e-8  # Prevent division by zero.
        # Compute the perturbation for coordinate d.
        r_d = (numerator / denominator) * tf.sign(w_flat[d])
        
        # Update x_current_flat at coordinate d.
        new_val = tf.clip_by_value(x_current_flat[d] + r_d, 0.0, 1.0)
        x_current_flat = tf.tensor_scatter_nd_update(x_current_flat, [[d]], [new_val])
        
        # Update r_total_flat: difference between new value and original value.
        new_perturb = new_val - x_orig_flat[d]
        r_total_flat = tf.tensor_scatter_nd_update(r_total_flat, [[d]], [new_perturb])
        
        # If the new value hits a bound, mark this coordinate as saturated.
        if new_val <= 0.0 or new_val >= 1.0:
            S.add(d)
        
        sparse_iter += 1

    # Reshape the final results back to the image shape.
    adv_image = tf.reshape(x_current_flat, image.shape)
    r_total = tf.reshape(r_total_flat, image.shape)
    
    return adv_image, r_total

