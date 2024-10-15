from diffusers import UNet2DConditionModel


def build_unet(size='normal', num_channels=3):
    if size == 'normal':
        print('Normal U-net')
        down_blocks = (
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        )
        up_blocks = (
            "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        )
        channels = (16, 32, 64, 128)
    elif size == 'small':
        print('Small U-net')
        down_blocks = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        )
        up_blocks = (
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        )
        channels = (16, 32, 64, 64)

    # Create a model
    model = UNet2DConditionModel(
        sample_size=64,  # the target image resolution
        in_channels=num_channels,  # the number of input channels, 3 for RGB images
        out_channels=num_channels,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        block_out_channels=channels,  # More channels -> more parameters
        norm_num_groups=1,  # number of groups for groupnorm
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        encoder_hid_dim_type='text_proj',
        encoder_hid_dim=512,
        mid_block_type='UNetMidBlock2DSimpleCrossAttn'
    )
    return model

if __name__ == '__main__':
    model = build_unet(size='normal')
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    print(model)
