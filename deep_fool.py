import tensorflow as tf
import numpy as np

def deepfool_attack(
    model, 
    img_input, 
    num_classes, 
    max_iter=20, 
    overshoot=0.02, 
    eps=1e-6,
    clip_min=0.0,
    clip_max=255.0,
    verbose=True
):
    """
    Perform DeepFool attack on a single image in the [0, 255] range.
    
    Args:
        model: TF/Keras model (must output logits, not probabilities).
        img_input: Input image tensor with shape (1, H, W, C) in [0, 255].
        num_classes: Number of output classes.
        max_iter: Maximum iterations (default: 50).
        overshoot: Perturbation safety factor (default: 0.02).
        eps: Small value to avoid division by zero.
        clip_min/clip_max: Pixel value bounds (default: 0, 255).
        verbose: Whether to print attack progress (default: True).
        
    Returns:
        Adversarial image tensor (same shape as img_input).
    """
    # Convert input to tensor and initialize variables
    img_input = tf.convert_to_tensor(img_input, dtype=tf.float32)
    adv_image = tf.Variable(img_input, trainable=True)
    original_image = tf.identity(img_input)
    total_perturbation = tf.Variable(tf.zeros_like(original_image), trainable=True)
    
    # Get original prediction
    original_logits = model(original_image)
    original_class = tf.argmax(original_logits[0]).numpy()
    
    if verbose:
        print(f"Initial prediction: Class {original_class}")
    
    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            logits = model(adv_image)
        
        current_class = tf.argmax(logits[0]).numpy()
        
        # Debug prints (before checking for success)
        if verbose:
            print(f"\nIteration {iteration}:")
            print(f"  Current class: {current_class}")
        
        # Check for misclassification
        if current_class != original_class:
            if verbose:
                print(f"\nAttack succeeded at iteration {iteration}!")
                print(f"Original class: {original_class} -> Adversarial class: {current_class}")
            break
        
        # Compute gradients of all logits w.r.t. input
        grads = tape.jacobian(logits, adv_image)  # Shape: (1, num_classes, 1, H, W, C)
        grads = tf.squeeze(grads, axis=[0, 2])    # Remove batch dim -> (num_classes, H, W, C)
        
        logits_diff = logits[0] - logits[0, original_class]
        min_perturb_norm = float('inf')
        best_perturb = None
        
        # Find minimal perturbation across all classes
        for k in range(num_classes):
            if k == original_class:
                continue
            
            w_k = grads[k] - grads[original_class]  # Shape: (H, W, C)
            f_k = logits_diff[k]
            
            norm_w_k = tf.norm(tf.reshape(w_k, [-1]))
            if norm_w_k < eps:
                continue
            
            # Compute perturbation for class k
            perturb_k = (tf.abs(f_k) / (norm_w_k**2 + eps)) * w_k
            perturb_k = tf.expand_dims(perturb_k, axis=0)  # Add batch dim -> (1, H, W, C)
            
            perturb_k_norm = tf.norm(perturb_k)
            if perturb_k_norm < min_perturb_norm:
                min_perturb_norm = perturb_k_norm
                best_perturb = perturb_k
        
        if best_perturb is None:
            if verbose:
                print("No valid perturbation found - terminating early")
            break
        
        # Update perturbation and adversarial image
        total_perturbation.assign_add(best_perturb)
        adv_image.assign(original_image + (1 + overshoot) * total_perturbation)
        adv_image.assign(tf.clip_by_value(adv_image, clip_min, clip_max))
    
    if verbose and current_class == original_class:
        print("\nAttack failed to misclassify within max iterations")
    
    return adv_image.numpy()