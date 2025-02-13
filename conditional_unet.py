from diffusers import UNet2DConditionModel


def build_unet(num_channels=3):
    """
    Creates a U-net 2D model
    Args:
        num_channels: int, number of input and output channels
    Returns:
        unet: UNet2DConditionModel, a U-net model configured for 2D image processing
    """
    # Define downsampling blocks
    down_blocks = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D"
    )
    # Define upsampling blocks
    up_blocks = (
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    )
    # Channel sizes
    channels = (16, 32, 64, 128)

    # Create model
    unet = UNet2DConditionModel(
        sample_size=64,  # the target image resolution
        in_channels=num_channels,  # the number of input channels, 3 for RGB images
        out_channels=num_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=channels,  # More channels -> more parameters
        norm_num_groups=1,  # number of groups for groupnorm
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        cross_attention_dim=512,
        mid_block_type='UNetMidBlock2DSimpleCrossAttn',
    )

    return unet


if __name__ == '__main__':
    unet = build_unet()
    print("Num params: ", sum(p.numel() for p in unet.parameters()))

