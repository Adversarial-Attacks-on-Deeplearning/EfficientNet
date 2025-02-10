import math

def calculate_padding(input_dim, kernel_size, stride, output_dim):
    """
    Calculate padding needed for a convolution operation based on output dimensions.
    
    Parameters:
    - input_dim (int): The input dimension (height or width).
    - kernel_size (int): The kernel size (height or width).
    - stride (int): The stride for the convolution.
    - output_dim (int): The desired output dimension.

    Returns:
    - padding_before (int): Padding to apply before the input.
    - padding_after (int): Padding to apply after the input.
    """
    # Calculate the total padding required
    padding_total = ((output_dim - 1) * stride + kernel_size - input_dim) 

    # If padding is negative, return zero padding
    if padding_total < 0:
        return 0, 0

    # Calculate the padding before and after
    padding_before = math.floor(padding_total / 2)
    padding_after = padding_total - padding_before

    return padding_before, padding_after

# Example usage:
input_size = 15  # Example input dimension (height or width)
kernel_size = 5  # Example kernel size
stride = 2       # Example stride
output_size = 8  # Desired output dimension

padding = calculate_padding(input_size, kernel_size, stride, output_size)
print(f"Padding needed: {padding}")


def compute_single_layer_output(initial_resolution, kernel_size, stride):
    """
    Computes the output resolution for a single layer without padding.

    Parameters:
    - initial_resolution: Tuple of (height, width) representing the input resolution.
    - kernel_size: Size of the kernel (int).
    - stride: Stride of the convolution (int).

    Returns:
    - int representing the output resolution size.
    """

    out = int((initial_resolution + stride - 1) / stride) 

    return out


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}


DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def bn_axis():
    return 3
