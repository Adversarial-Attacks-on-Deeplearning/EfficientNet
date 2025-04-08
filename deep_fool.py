import tensorflow as tf
import numpy as np

def deepfool_attack(model, image, max_iter=50, overshoot=5):
    """
    DeepFool attack implementation for image classification using TensorFlow.
    
    This version is designed for an EfficientNet-based classification model that accepts
    images with pixel values in [0, 255]. The attack iteratively perturbs the input image
    until the predicted class changes.
    
    Args:
      model: tf.keras.Model that outputs logits.
      image: tf.Tensor or numpy array with shape (1, H, W, C) containing the input image,
             with pixel values in the range [0, 255].
      max_iter: Maximum number of iterations for the attack.
      overshoot: Maximum allowed perturbation per pixel (applied per update).
      
    Returns:
      pert_image: The adversarially perturbed image as a numpy array.
      r_total: The total accumulated perturbation as a numpy array.
    """
    # Ensure image is a tf.Tensor with dtype float32.
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Get original prediction (assume batch size = 1)
    output = model(image, training=False)
    original_class = tf.argmax(output, axis=1)  # integer tensor of shape (1,)
    
    # Create a variable for the perturbed image
    pert_image = tf.Variable(image)
    
    # Initialize the total perturbation to zeros
    r_total = tf.zeros_like(image, dtype=tf.float32)
    
    loop_i = 0
    while loop_i < max_iter:
        with tf.GradientTape() as tape:
            tape.watch(pert_image)
            # Get the output logits for the current perturbed image.
            output = model(pert_image, training=False)
            
            # For a single image, extract the logit for the original (correct) class.
            correct_logit = output[0, original_class[0]]
            # Define loss as negative of the correct class logit so that minimizing loss
            # will drive that logit down.
            loss = -correct_logit
        
        # Compute the gradient of the loss with respect to the input image.
        grad = tape.gradient(loss, pert_image)
        
        # Normalize the gradient to obtain a unit vector.
        norm_grad = tf.norm(grad) + 1e-8  # small epsilon to avoid division by zero
        w = grad / norm_grad
        
        # Compute the incremental perturbation. The scaling factor (loss + 1e-4)
        # determines the step size.
        r_i = (loss + 1e-4) * w
        
        # Accumulate the perturbation and clip it so that per-pixel perturbation
        # does not exceed the overshoot value.
        r_total = tf.clip_by_value(r_total + r_i, -overshoot, overshoot)
        
        # Update the perturbed image and ensure pixel values remain in [0, 255].
        pert_image.assign(tf.clip_by_value(image + r_total, 0.0, 255.0))
        
        # Check if the predicted class has changed.
        new_output = model(pert_image, training=False)
        new_class = tf.argmax(new_output, axis=1)
        if new_class.numpy()[0] != original_class.numpy()[0]:
            break
        
        loop_i += 1
        print(f"Iteration {loop_i}: Class predicted: {new_class.numpy()[0]}, ")

    return pert_image.numpy(), r_total.numpy()
