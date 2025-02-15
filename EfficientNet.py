import tensorflow as tf
import myUtils
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,GlobalAveragePooling2D
import math


def mbConv_block(input, input_channels, output_channel, t, s, kernel_size=3, drop_rate=0.2, block_name="Block", block_num=0, survival_prob=0.8, output_resolution=None):
    """
    Constructs an MBConv block using the functional API with an SE block and L2 regularization.

    Parameters:
    - input: Input tensor.
    - input_channels: Number of input channels.
    - output_channel: Number of output channels.
    - t: Expansion factor.
    - s: Stride for depthwise convolution.
    - kernel_size: Kernel size for depthwise convolution.
    - drop_rate: Dropout rate.
    - block_name: Name of the block.
    - block_num: Block number (useful for debugging).
    - survival_prob: DropConnect survival probability.
    - output_resolution: Output resolution for padding calculation.

    Returns:
    - Output tensor from the MBConv block.
    """
    bn_axis = 3
    # Block A---------------------------------------------------------------------------------------------------------------

    # Expansion
    expanded_filters = input_channels * t
    padding = 'same'

    if t > 1:
        # Expansion Convolution
        x = Conv2D(
            expanded_filters,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=myUtils.CONV_KERNEL_INITIALIZER,
            name=f'{block_name}_Expansion_Conv'
        )(input)
        x = tf.keras.layers.BatchNormalization(name=f'{block_name}_Expansion_BN',axis=bn_axis)(x)
        x = tf.keras.activations.swish(x)  # Swish activation, no name needed
        se_ratio = (1 / 24)
        if s == 2:
            pad = myUtils.calculate_padding(input_dim=input.shape[1], kernel_size=kernel_size, stride=s, output_dim=output_resolution)
            x = tf.keras.layers.ZeroPadding2D(padding=(pad, pad), name=f'{block_name}_Zero_Padding')(x)
            padding = 'valid'

    # Depthwise Convolution
    if t == 1:
        se_ratio = (0.25)
        x = input
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        padding=padding,
        strides=s,
        use_bias=False,
        depthwise_initializer=myUtils.CONV_KERNEL_INITIALIZER,
        name=f'{block_name}_Depthwise_Conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f'{block_name}_Depthwise_BN',axis=bn_axis)(x)
    x = tf.keras.activations.swish(x)  # Swish activation, no name needed

    # Squeeze-and-Excitation (SE)
    se = GlobalAveragePooling2D(name=f'{block_name}_SE_Global_Avg_Pool')(x)
    se = tf.keras.layers.Reshape((1, 1, expanded_filters), name=f'{block_name}_SE_Reshape')(se)
    se_filters = max(1, int(expanded_filters * se_ratio))
    se = Conv2D(
        se_filters,
        kernel_size=1,
        activation="swish",
        kernel_initializer=myUtils.CONV_KERNEL_INITIALIZER,
        name=f'{block_name}_SE_Conv1'
    )(se)
    se = Conv2D(
        expanded_filters,
        kernel_size=1,
        activation="sigmoid",
        kernel_initializer=myUtils.CONV_KERNEL_INITIALIZER,
        name=f'{block_name}_SE_Conv2'
    )(se)
    x = tf.keras.layers.Multiply(name=f'{block_name}_SE_Multiply')([x, se])  # Multiply the input with the SE output

    # Projection Convolution
    x = Conv2D(
        output_channel,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_initializer=myUtils.CONV_KERNEL_INITIALIZER,
        name=f'{block_name}_Projection_Conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f'{block_name}_Projection_BN',axis=bn_axis)(x)


    # Skip connection (Residual connection)
    if s == 1 and input_channels == output_channel:
        x = Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=f'{block_name}_Dropout')(x)
        x = tf.keras.layers.Add(name=f'{block_name}_Skip_Connection')([input, x])

    return x


def scaledResolution(phi):
    """
    Scaled Resolution function for EfficientNetB0-B7.
    """
    cases = {
        0: 224,
        1: 240,
        2: 260,
        3: 300,
        4: 380,
        5: 456,
        6: 528,
        7: 600
    }
    return cases[phi]


def round_filters(filters, width_coefficient):
    filters *= width_coefficient
    return max(8, int(filters + 4) // 8 * 8)


def dropOut_rate(phi):
    """
    Dropout rate for EfficientNetB0-B7.
    """
    cases = {
        0: 0.2,
        1: 0.2,
        2: 0.3,
        3: 0.3,
        4: 0.4,
        5: 0.4,
        6: 0.5,
        7: 0.5
    }
    return cases[phi]



def efficientNet(input_shape=(224,224,3), num_classes=1000, phi=1):
    alpha = 1.2
    beta = 1.1
    gamma = 1.15
    bn_axis = 3

    depth = alpha ** phi
    width = beta ** phi
    scaled_resolution = scaledResolution(phi)
    scaled_input_shape = (scaled_resolution, scaled_resolution, input_shape[2])

    inputs = tf.keras.layers.Input(shape=scaled_input_shape, name='Input_Layer')
    # Preprocessing Layers
    x = tf.keras.layers.Normalization(name='Normalization')(inputs)
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1, name='Rescaling_1')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='ZeroPadding')(x)

    # Stem
    scaled_filters = round_filters(32, width)
    x = Conv2D(scaled_filters, 3, strides=2, padding='valid', use_bias=False, name='Stem_Conv')(x)
    x = tf.keras.layers.BatchNormalization(name='Stem_BN',axis=bn_axis)(x)
    x = tf.keras.activations.swish(x)

    base_blocks = [
        (32, 16, 1, 1, math.floor(1 * depth), 3),
        (16, 24, 6, 2, math.floor(2 * depth), 3),
        (24, 40, 6, 2, math.floor(2 * depth), 5),
        (40, 80, 6, 2, math.floor(3 * depth), 3),
        (80, 112, 6, 2, math.floor(3 * depth), 5),
        (112, 192, 6, 2, math.floor(4 * depth), 5),
        (192, 320, 6, 1, math.floor(1 * depth), 3),
    ]
    scaled_blocks = []
    for (in_ch, out_ch, t, s, repeats, kernel) in base_blocks:
        scaled_in = round_filters(in_ch, width)
        scaled_out = round_filters(out_ch, width)
        scaled_blocks.append((scaled_in, scaled_out, t, s, repeats, kernel))
    blocks_args = scaled_blocks
    block_idx = 0
# MBConv Blocks
    for stage_idx, (input_channels, output_channels, t, s, repeats, kernel) in enumerate(blocks_args):
        output_shape=myUtils.compute_single_layer_output(initial_resolution=x.shape[1], kernel_size=kernel, stride=s)
    # For subsequent blocks in the stage (e.g., Stage_1_Block_2_b, Stage_1_Block_3_c, etc.)
        for block_idx in range(0, repeats+1):
            # Name for the first block in each stage (e.g., Stage_1_Block_1_a)
            x = mbConv_block(
        x,
        input_channels=input_channels if block_idx == 0 else output_channels,
        output_channel=output_channels,
        t=t,
        s=s if block_idx == 0 else 1,
        kernel_size=kernel,
        block_name=f'Block{stage_idx+1}{chr(97+block_idx)}',
        block_num=block_idx,
        output_resolution=output_shape
        )


    # Head

    x = Conv2D(1280, 1, padding='same', use_bias=False, name=f'Head_Conv_{stage_idx+1}',kernel_initializer=myUtils.CONV_KERNEL_INITIALIZER)(x)
    x = tf.keras.layers.BatchNormalization(name=f'Head_BN_{stage_idx+1}',axis=bn_axis)(x)
    x = tf.keras.activations.swish(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='Global_Avg_Pool')(x)
    drop_rate = dropOut_rate(phi)
    x = Dropout(drop_rate, name='Dropout')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='Output_Dense',kernel_initializer=myUtils.DENSE_KERNEL_INITIALIZER)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='EfficientNet_Model')

    return model
