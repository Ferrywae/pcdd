(.venv) PS D:\cobaa\FourLLIE-main>  python test.py -opt options/test/LOLv2_real.yml
export CUDA_VISIBLE_DEVICES=
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
D:\cobaa\FourLLIE-main\.venv\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
D:\cobaa\FourLLIE-main\.venv\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: D:\cobaa\FourLLIE-main\.venv\lib\site-packages\lpips\weights\v0.1\alex.pth
INFO:base:Network G structure: FourLLIE, with parameters: 119,446
INFO:base:FourLLIE(
  (AmpNet): Sequential(
    (0): AmplitudeNet_skip(
      (conv0): Sequential(
        (0): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
        (1): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (conv1): ProcessBlock(
        (spatial_process): SpaBlock(
          (block): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (frequency_process): FreBlock(
          (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          (process1): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
          (process2): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv2): ProcessBlock(
        (spatial_process): SpaBlock(
          (block): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (frequency_process): FreBlock(
          (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          (process1): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
          (process2): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv3): ProcessBlock(
        (spatial_process): SpaBlock(
          (block): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (frequency_process): FreBlock(
          (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          (process1): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
          (process2): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv4): Sequential(
        (0): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv5): Sequential(
        (0): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (convout): Sequential(
        (0): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (1): Sigmoid()
  )
  (conv_first_1): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_first_2): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv_first_3): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (feature_extraction): Sequential(
    (0): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (recon_trunk): Sequential(
    (0): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (upconv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (upconv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pixel_shuffle): PixelShuffle(upscale_factor=2)
  (HRconv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_last): Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
  (transformer): SFNet(
    (conv1): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv2): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv3): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv4): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv5): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (recon_trunk_light): Sequential(
    (0): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (2): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (3): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (4): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (5): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
)
INFO:base:Loading model for G [./pre-trained/nikon.pth] ...
INFO:base:Model [enhancement_model] is created.
mkdir finish
25-10-20 14:20:18.206 - INFO: Dataset [ll_dataset - test] is created.
INFO:base:Dataset [ll_dataset - test] is created.
[                                                  ] 0/30, elapsed: 0s, ETA:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 30/30, 1.9 task/s, elapsed: 16s, ETA:     0s
Test 0030 - ['0/43']
25-10-20 14:20:35.115 - INFO: # Validation # PSNR: 15.302718: 0001: 15.373255 0002: 16.695826 0003: 14.300511 0004: 12.493755 0005: 12.027312 0006: 13.902086 0007: 17.816109 0008: 17.142409 0009: 15.975342 0010: 16.297274 0011: 15.632355 0012: 14.490138 0013: 15.836133 0014: 16.319133 0015: 17.291744 0016: 17.405661 0017: 12.036561 0018: 18.781743 0019: 15.361421 0020: 12.884239 0021: 14.740788 0022: 16.126705 0023: 16.089986 0024: 13.707754 0025: 16.876488 0026: 14.001173 0027: 15.046475 0028: 15.567123 0029: 12.629455 0030: 16.232584
INFO:base:# Validation # PSNR: 15.302718: 0001: 15.373255 0002: 16.695826 0003: 14.300511 0004: 12.493755 0005: 12.027312 0006: 13.902086 0007: 17.816109 0008: 17.142409 0009: 15.975342 0010: 16.297274 0011: 15.632355 0012: 14.490138 0013: 15.836133 0014: 16.319133 0015: 17.291744 0016: 17.405661 0017: 12.036561 0018: 18.781743 0019: 15.361421 0020: 12.884239 0021: 14.740788 0022: 16.126705 0023: 16.089986 0024: 13.707754 0025: 16.876488 0026: 14.001173 0027: 15.046475 0028: 15.567123 0029: 12.629455 0030: 16.232584
25-10-20 14:20:35.116 - INFO: # Validation # SSIM: 0.557847: 0001: 0.558267 0002: 0.578666 0003: 0.567259 0004: 0.526717 0005: 0.517276 0006: 0.528972 0007: 0.574628 0008: 0.582559 0009: 0.569636 0010: 0.562889 0011: 0.561837 0012: 0.553927 0013: 0.555898 0014: 0.581541 0015: 0.573449 0016: 0.589362 0017: 0.532288 0018: 0.596604 0019: 0.566180 0020: 0.536989 0021: 0.570662 0022: 0.554777 0023: 0.570186 0024: 0.526406 0025: 0.581657 0026: 0.534155 0027: 0.554206 0028: 0.551533 0029: 0.542767 0030: 0.534114
INFO:base:# Validation # SSIM: 0.557847: 0001: 0.558267 0002: 0.578666 0003: 0.567259 0004: 0.526717 0005: 0.517276 0006: 0.528972 0007: 0.574628 0008: 0.582559 0009: 0.569636 0010: 0.562889 0011: 0.561837 0012: 0.553927 0013: 0.555898 0014: 0.581541 0015: 0.573449 0016: 0.589362 0017: 0.532288 0018: 0.596604 0019: 0.566180 0020: 0.536989 0021: 0.570662 0022: 0.554777 0023: 0.570186 0024: 0.526406 0025: 0.581657 0026: 0.534155 0027: 0.554206 0028: 0.551533 0029: 0.542767 0030: 0.534114  
25-10-20 14:20:35.116 - INFO: # Validation # LPIPS: 0.350518: 0001: 0.328224 0002: 0.298593 0003: 0.343670 0004: 0.497181 0005: 0.425949 0006: 0.456787 0007: 0.298057 0008: 0.282220 0009: 0.331394 0010: 0.320958 0011: 0.320476 0012: 0.383382 0013: 0.255888 0014: 0.264187 0015: 0.271895 0016: 0.302679 0017: 0.504359 0018: 0.258925 0019: 0.359624 0020: 0.415740 0021: 0.401116 0022: 0.352260 0023: 0.327156 0024: 0.398712 0025: 0.268063 0026: 0.418400 0027: 0.346482 0028: 0.331571 0029: 0.456331 0030: 0.295271
INFO:base:# Validation # LPIPS: 0.350518: 0001: 0.328224 0002: 0.298593 0003: 0.343670 0004: 0.497181 0005: 0.425949 0006: 0.456787 0007: 0.298057 0008: 0.282220 0009: 0.331394 0010: 0.320958 0011: 0.320476 0012: 0.383382 0013: 0.255888 0014: 0.264187 0015: 0.271895 0016: 0.302679 0017: 0.504359 0018: 0.258925 0019: 0.359624 0020: 0.415740 0021: 0.401116 0022: 0.352260 0023: 0.327156 0024: 0.398712 0025: 0.268063 0026: 0.418400 0027: 0.346482 0028: 0.331571 0029: 0.456331 0030: 0.295271 
PSNR_all: 15.302717998989545
SSIM_all: 0.557846930088873
LPIPS_all: 0.3505182941754659
(.venv) PS D:\cobaa\FourLLIE-main>  python test.py -opt options/test/LOLv2_real.yml
export CUDA_VISIBLE_DEVICES=
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
D:\cobaa\FourLLIE-main\.venv\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
D:\cobaa\FourLLIE-main\.venv\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: D:\cobaa\FourLLIE-main\.venv\lib\site-packages\lpips\weights\v0.1\alex.pth
INFO:base:Network G structure: FourLLIE, with parameters: 119,446
INFO:base:FourLLIE(
  (AmpNet): Sequential(
    (0): AmplitudeNet_skip(
      (conv0): Sequential(
        (0): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
        (1): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (conv1): ProcessBlock(
        (spatial_process): SpaBlock(
          (block): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (frequency_process): FreBlock(
          (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          (process1): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
          (process2): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv2): ProcessBlock(
        (spatial_process): SpaBlock(
          (block): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (frequency_process): FreBlock(
          (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          (process1): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
          (process2): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv3): ProcessBlock(
        (spatial_process): SpaBlock(
          (block): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (frequency_process): FreBlock(
          (fpre): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          (process1): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
          (process2): Sequential(
            (0): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.1, inplace=True)
            (2): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cat): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv4): Sequential(
        (0): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv5): Sequential(
        (0): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
      )
      (convout): Sequential(
        (0): ProcessBlock(
          (spatial_process): SpaBlock(
            (block): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (frequency_process): FreBlock(
            (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            (process1): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
            (process2): Sequential(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (1): LeakyReLU(negative_slope=0.1, inplace=True)
              (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (cat): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (1): Sigmoid()
  )
  (conv_first_1): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_first_2): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv_first_3): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (feature_extraction): Sequential(
    (0): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (recon_trunk): Sequential(
    (0): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (upconv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (upconv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pixel_shuffle): PixelShuffle(upscale_factor=2)
  (HRconv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_last): Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
  (transformer): SFNet(
    (conv1): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv2): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv3): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv4): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
    (conv5): ProcessBlock(
      (spatial_process): Identity()
      (frequency_process): FreBlock(
        (fpre): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        (process1): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (process2): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): LeakyReLU(negative_slope=0.1, inplace=True)
          (2): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cat): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (recon_trunk_light): Sequential(
    (0): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (2): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (3): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (4): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (5): ResidualBlock_noBN(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
)
INFO:base:Loading model for G [./pre-trained/nikon.pth] ...
INFO:base:Model [enhancement_model] is created.
mkdir finish
25-10-20 14:32:17.895 - INFO: Dataset [ll_dataset - test] is created.
INFO:base:Dataset [ll_dataset - test] is created.
[                                                  ] 0/30, elapsed: 0s, ETA:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 30/30, 2.0 task/s, elapsed: 15s, ETA:     0s
Test 0030 - ['0/43']
25-10-20 14:32:33.633 - INFO: # Validation # PSNR: 15.302718: 0001: 15.373255 0002: 16.695826 0003: 14.300511 0004: 12.493755 0005: 12.027312 0006: 13.902086 0007: 17.816109 0008: 17.142409 0009: 15.975342 0010: 16.297274 0011: 15.632355 0012: 14.490138 0013: 15.836133 0014: 16.319133 0015: 17.291744 0016: 17.405661 0017: 12.036561 0018: 18.781743 0019: 15.361421 0020: 12.884239 0021: 14.740788 0022: 16.126705 0023: 16.089986 0024: 13.707754 0025: 16.876488 0026: 14.001173 0027: 15.046475 0028: 15.567123 0029: 12.629455 0030: 16.232584
INFO:base:# Validation # PSNR: 15.302718: 0001: 15.373255 0002: 16.695826 0003: 14.300511 0004: 12.493755 0005: 12.027312 0006: 13.902086 0007: 17.816109 0008: 17.142409 0009: 15.975342 0010: 16.297274 0011: 15.632355 0012: 14.490138 0013: 15.836133 0014: 16.319133 0015: 17.291744 0016: 17.405661 0017: 12.036561 0018: 18.781743 0019: 15.361421 0020: 12.884239 0021: 14.740788 0022: 16.126705 0023: 16.089986 0024: 13.707754 0025: 16.876488 0026: 14.001173 0027: 15.046475 0028: 15.567123 0029: 12.629455 0030: 16.232584
25-10-20 14:32:33.638 - INFO: # Validation # SSIM: 0.557847: 0001: 0.558267 0002: 0.578666 0003: 0.567259 0004: 0.526717 0005: 0.517276 0006: 0.528972 0007: 0.574628 0008: 0.582559 0009: 0.569636 0010: 0.562889 0011: 0.561837 0012: 0.553927 0013: 0.555898 0014: 0.581541 0015: 0.573449 0016: 0.589362 0017: 0.532288 0018: 0.596604 0019: 0.566180 0020: 0.536989 0021: 0.570662 0022: 0.554777 0023: 0.570186 0024: 0.526406 0025: 0.581657 0026: 0.534155 0027: 0.554206 0028: 0.551533 0029: 0.542767 0030: 0.534114
INFO:base:# Validation # SSIM: 0.557847: 0001: 0.558267 0002: 0.578666 0003: 0.567259 0004: 0.526717 0005: 0.517276 0006: 0.528972 0007: 0.574628 0008: 0.582559 0009: 0.569636 0010: 0.562889 0011: 0.561837 0012: 0.553927 0013: 0.555898 0014: 0.581541 0015: 0.573449 0016: 0.589362 0017: 0.532288 0018: 0.596604 0019: 0.566180 0020: 0.536989 0021: 0.570662 0022: 0.554777 0023: 0.570186 0024: 0.526406 0025: 0.581657 0026: 0.534155 0027: 0.554206 0028: 0.551533 0029: 0.542767 0030: 0.534114  
25-10-20 14:32:33.638 - INFO: # Validation # LPIPS: 0.350518: 0001: 0.328224 0002: 0.298593 0003: 0.343670 0004: 0.497181 0005: 0.425949 0006: 0.456787 0007: 0.298057 0008: 0.282220 0009: 0.331394 0010: 0.320958 0011: 0.320476 0012: 0.383382 0013: 0.255888 0014: 0.264187 0015: 0.271895 0016: 0.302679 0017: 0.504359 0018: 0.258925 0019: 0.359624 0020: 0.415740 0021: 0.401116 0022: 0.352260 0023: 0.327156 0024: 0.398712 0025: 0.268063 0026: 0.418400 0027: 0.346482 0028: 0.331571 0029: 0.456331 0030: 0.295271
INFO:base:# Validation # LPIPS: 0.350518: 0001: 0.328224 0002: 0.298593 0003: 0.343670 0004: 0.497181 0005: 0.425949 0006: 0.456787 0007: 0.298057 0008: 0.282220 0009: 0.331394 0010: 0.320958 0011: 0.320476 0012: 0.383382 0013: 0.255888 0014: 0.264187 0015: 0.271895 0016: 0.302679 0017: 0.504359 0018: 0.258925 0019: 0.359624 0020: 0.415740 0021: 0.401116 0022: 0.352260 0023: 0.327156 0024: 0.398712 0025: 0.268063 0026: 0.418400 0027: 0.346482 0028: 0.331571 0029: 0.456331 0030: 0.295271 
PSNR_all: 15.302717998989545
SSIM_all: 0.557846930088873
LPIPS_all: 0.3505182941754659
(.venv) PS D:\cobaa\FourLLIE-main> git push
fatal: not a git repository (or any of the parent directories): .git
(.venv) PS D:\cobaa\FourLLIE-main> cd D:\       
(.venv) PS D:\> 
(.venv) PS D:\> # 0) (opsional) set identitas git sekali saja
>> git config --global user.name  "ferrywae"
>> git config --global user.email "emailmu@domain.com"
>> git config --global core.autocrlf true
>> git config --global credential.helper manager
>>
>> # 1) masuk ke folder proyek
>> cd D:\cobaa\FourLLIE-main
>>
>> # 2) inisialisasi repo
>> git init
>>
>> # 3) buat .gitignore biar file besar ga ikut
>> @"
>> __pycache__/
>> *.pyc
>> .venv/
>> .env
>> results/
>> data_test/
>> pre-trained/
>> *.pth
>> *.pt
>> .vscode/
>> .DS_Store
>> Thumbs.db
>> "@ | Set-Content .gitignore
>>
>> # 4) (opsional) README
>> @"
>> # pcd
>> FourLLIE + script test & evaluasi (PSNR/SSIM/LPIPS).
>> "@ | Set-Content README.md
>>
>> # 5) commit awal
>> git add -A
>> git commit -m "init: setup FourLLIE + test & metrics"
>>
>> # 6) set remote ke repo kamu
>> git branch -M main
>> git remote add origin https://github.com/ferrywae/pcd.git
>> # kalau sebelumnya sudah ada remote, pakai:
>> # git remote set-url origi
>>
Initialized empty Git repository in D:/cobaa/FourLLIE-main/.git/
warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'LICENSE', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '_cek_channel.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'data/LL_dataset.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'data/__init__.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'data/data_sampler.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'data/util.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/__init__.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/archs/FourLLIE.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/archs/SFBlock.py', LF will be replaced by CRLF the next time Git touches it        
warning: in the working copy of 'models/archs/arch_util.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/base_model.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/enhancement_model.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/loss.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/lr_scheduler.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/networks.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'options/options.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'options/test/LOLv2_real.yml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'options/train/LOLv2_real.yml', LF will be replaced by CRLF the next time Git touches it   
warning: in the working copy of 'requirements.txt', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'test.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tools/download_lolv2_synth.py', LF will be replaced by CRLF the next time Git touches it  
warning: in the working copy of 'tools/download_lolv2_test.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'train.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/util.py', LF will be replaced by CRLF the next time Git touches it
[master (root-commit) 5afa2b5] init: setup FourLLIE + test & metrics
 31 files changed, 3105 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 LICENSE
 create mode 100644 README.md
 create mode 100644 _cek_channel.py
 create mode 100644 data/LL_dataset.py
 create mode 100644 data/__init__.py
 create mode 100644 data/data_sampler.py
 create mode 100644 data/util.py
 create mode 100644 figs/pipeline.png
 create mode 100644 models/__init__.py
 create mode 100644 models/archs/FourLLIE.py
 create mode 100644 models/archs/SFBlock.py
 create mode 100644 models/archs/__init__.py
 create mode 100644 models/archs/arch_util.py
 create mode 100644 models/base_model.py
 create mode 100644 models/enhancement_model.py
 create mode 100644 models/loss.py
 create mode 100644 models/lr_scheduler.py
 create mode 100644 models/networks.py
 create mode 100644 options/__init__.py
 create mode 100644 options/options.py
 create mode 100644 options/test/LOLv2_real.yml
 create mode 100644 options/train/LOLv2_real.yml
 create mode 100644 requirements.txt
 create mode 100644 test.py
 create mode 100644 tools/download_lolv2_synth.py
 create mode 100644 tools/download_lolv2_test.py
 create mode 100644 tools/make_dummy_pairs.py
 create mode 100644 train.py
 create mode 100644 utils/__init__.py
 create mode 100644 utils/util.py
(.venv) PS D:\cobaa\FourLLIE-main> git remote add origin https://github.com/Ferrywae/pcd.git
>> git branch -M main
>> git push -u origin main
error: remote origin already exists.
Enumerating objects: 40, done.
Counting objects: 100% (40/40), done.
Delta compression using up to 8 threads
Compressing objects: 100% (36/36), done.
Writing objects: 100% (40/40), 149.45 KiB | 5.75 MiB/s, done.
Total 40 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
remote: This repository moved. Please use the new location:
remote:   https://github.com/Ferrywae/pcd.git
To https://github.com/ferrywae/pcd.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
(.venv) PS D:\cobaa\FourLLIE-main> git pull --rebase origin main
>> git push -u origin main
>>
From https://github.com/ferrywae/pcd
 * branch            main       -> FETCH_HEAD
Already up to date.
branch 'main' set up to track 'origin/main'.
Everything up-to-date
(.venv) PS D:\cobaa\FourLLIE-main> python --version
Python 3.10.11
(.venv) PS D:\cobaa\FourLLIE-main> conda
conda : The term 'conda' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling 
of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ conda
+ ~~~~~
    + CategoryInfo          : ObjectNotFound: (conda:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(.venv) PS D:\cobaa\FourLLIE-main> # 1) buat & aktifkan env conda
>> conda create -n pcd python=3.10 -y
>> conda activate pcd
>>
>> # 2a) PyTorch CPU saja
>> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
>>
>> # (atau) 2b) PyTorch GPU (NVIDIA, kalau drivernya cocok)
>> # conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
>>
>> # 3) lib lain yang dipakai proyekmu
>> pip install lpips opencv-python scikit-image pyyaml tqdm
>>
>> # 4) cek cepat
>> python -c "import torch,cv2,lpips; print('torch=',torch.__version__,'opencv=',cv2.__version__)"
>>
conda : The term 'conda' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling 
of the name, or if a path was included, verify that the path is correct and try again.
At line:2 char:1
+ conda create -n pcd python=3.10 -y
+ ~~~~~
    + CategoryInfo          : ObjectNotFound: (conda:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

conda : The term 'conda' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling 
of the name, or if a path was included, verify that the path is correct and try again.
At line:3 char:1
+ conda activate pcd
+ ~~~~~
    + CategoryInfo          : ObjectNotFound: (conda:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

Looking in indexes: https://download.pytorch.org/whl/cpu
Requirement already satisfied: torch in d:\cobaa\fourllie-main\.venv\lib\site-packages (2.9.0+cpu)
Requirement already satisfied: torchvision in d:\cobaa\fourllie-main\.venv\lib\site-packages (0.24.0+cpu)
Requirement already satisfied: torchaudio in d:\cobaa\fourllie-main\.venv\lib\site-packages (2.9.0+cpu)
Requirement already satisfied: filelock in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch) (3.19.1)
Requirement already satisfied: typing-extensions>=4.10.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch) (4.15.0)    
Requirement already satisfied: sympy>=1.13.3 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch) (3.3)
Requirement already satisfied: jinja2 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch) (2025.9.0)
Requirement already satisfied: numpy in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torchvision) (2.1.2)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torchvision) (11.3.0)  
Requirement already satisfied: mpmath<1.4,>=1.1.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from jinja2->torch) (2.1.5)
Requirement already satisfied: lpips in d:\cobaa\fourllie-main\.venv\lib\site-packages (0.1.4)
Requirement already satisfied: opencv-python in d:\cobaa\fourllie-main\.venv\lib\site-packages (4.12.0.88)
Requirement already satisfied: scikit-image in d:\cobaa\fourllie-main\.venv\lib\site-packages (0.25.2)
Requirement already satisfied: pyyaml in d:\cobaa\fourllie-main\.venv\lib\site-packages (6.0.3)
Requirement already satisfied: tqdm in d:\cobaa\fourllie-main\.venv\lib\site-packages (4.67.1)
Requirement already satisfied: torch>=0.4.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from lpips) (2.9.0+cpu)
Requirement already satisfied: torchvision>=0.2.1 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from lpips) (0.24.0+cpu)       
Requirement already satisfied: numpy>=1.14.3 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from lpips) (2.1.2)
Requirement already satisfied: scipy>=1.0.1 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from lpips) (1.15.3)
Requirement already satisfied: networkx>=3.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from scikit-image) (3.3)
Requirement already satisfied: pillow>=10.1 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from scikit-image) (11.3.0)
Requirement already satisfied: imageio!=2.35.0,>=2.33 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from scikit-image) (2.37.0)
Requirement already satisfied: tifffile>=2022.8.12 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from scikit-image) (2025.5.10)
Requirement already satisfied: packaging>=21 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from scikit-image) (25.0)
Requirement already satisfied: lazy-loader>=0.4 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from scikit-image) (0.4)
Requirement already satisfied: colorama in d:\cobaa\fourllie-main\.venv\lib\site-packages (from tqdm) (0.4.6)
Requirement already satisfied: filelock in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch>=0.4.0->lpips) (3.19.1)
Requirement already satisfied: typing-extensions>=4.10.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch>=0.4.0->lpips) (4.15.0)
Requirement already satisfied: sympy>=1.13.3 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch>=0.4.0->lpips) (1.14.0)  
Requirement already satisfied: jinja2 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch>=0.4.0->lpips) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from torch>=0.4.0->lpips) (2025.9.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from sympy>=1.13.3->torch>=0.4.0->lpips) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in d:\cobaa\fourllie-main\.venv\lib\site-packages (from jinja2->torch>=0.4.0->lpips) (2.1.5)
torch= 2.9.0+cpu opencv= 4.12.0
(.venv) PS D:\cobaa\FourLLIE-main> conda
conda : The term 'conda' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling 
of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ conda
+ ~~~~~
    + CategoryInfo          : ObjectNotFound: (conda:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(.venv) PS D:\cobaa\FourLLIE-main> # 1) buat & aktifkan env conda
>> conda create -n pcd python=3.10 -y
>> conda activate pcd
>>
>> # 2a) PyTorch CPU saja
>> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
>>
>> # (atau) 2b) PyTorch GPU (NVIDIA, kalau drivernya cocok)
>> # conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
>>
>> # 3) lib lain yang dipakai proyekmu
>> pip install lpips opencv-python scikit-image pyyaml tqdm
>>
>> # 4) cek cepat
>> python -c "import torch,cv2,lpips; print('torch=',torch.__version__,'opencv=',cv2.__version__)"
>>
(.venv) PS D:\cobaa\FourLLIE-main> # 1) buat & aktifkan env conda
>> conda create -n pcd python=3.10 -y
>> conda activate pcd
>> 
>> # 2a) PyTorch CPU saja
>> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
>> 
>> # (atau) 2b) PyTorch GPU (NVIDIA, kalau drivernya cocok)
>> # conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
>> 
>> # 3) lib lain yang dipakai proyekmu
>> pip install lpips opencv-python scikit-image pyyaml tqdm
>>
>> # 4) cek cepat
>> python -c "import torch,cv2,lpips; print('torch=',torch.__version__,'opencv=',cv2.__version__)"
>>
(.venv) PS D:\cobaa\FourLLIE-main> # 1) buat & aktifkan env conda
>> conda create -n pcd python=3.10 -y
>> conda activate pcd
>> 
>> # 2a) PyTorch CPU saja
>> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
>>
>> # (atau) 2b) PyTorch GPU (NVIDIA, kalau drivernya cocok)
>> # conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
>>
>> # 3) lib lain yang dipakai proyekmu
>> pip install lpips opencv-python scikit-image pyyaml tqdm
>>
>> # 4) cek cepat
>> python -c "import torch,cv2,lpips; print('torch=',torch.__version__,'opencv=',cv2.__version__)"
>>



