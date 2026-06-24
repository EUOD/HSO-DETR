class FSConv(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.c1 = c1

        self.conv1 = Conv(c1, 2 * c1, 1, g=c1)

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv2 = Conv(c1, c2, k, 2, g=math.gcd(c1, c2))

        self.conv3 = Conv(
            c1 * 3,
            c2,
            3,
            d=1,
            g=math.gcd(c1 * 3, c2)
        )

        self.se = SEAttention(c2)

        self.conv4 = Conv(
            c1,
            c2,
            3,
            g=math.gcd(c1, c2)
        )

        self.conv5 = Conv(2 * c2, c2, 1)

    def forward(self, x):
        # x: [B, c1, H, W]

        x0 = self.conv1(x)
        # x0: [B, 2 * c1, H, W]

        x1, x2 = torch.split(x0, self.c1, dim=1)
        # x1: [B, c1, H, W]
        # x2: [B, c1, H, W]

        conv_spatial = self.conv2(x1)
        # conv_spatial: [B, c2, H/2, W/2]

        yL, yH = self.wt(x2)
        # yL: [B, c1, H/2, W/2]
        # yH[0]: [B, c1, 3, H/2, W/2]

        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        # each: [B, c1, H/2, W/2]

        high_frequency_fused = torch.cat([y_HL, y_LH, y_HH], dim=1)
        # high_frequency_fused: [B, 3 * c1, H/2, W/2]

        high_frequency_fused_output = self.conv3(high_frequency_fused)
        # high_frequency_fused_output: [B, c2, H/2, W/2]

        high_frequency_fused_output = self.se(high_frequency_fused_output)

        low_frequency_fused_output = self.conv4(yL)
        # low_frequency_fused_output: [B, c2, H/2, W/2]

        spatial_output = conv_spatial * high_frequency_fused_output
        # spatial_output: [B, c2, H/2, W/2]

        fused = torch.cat([spatial_output, low_frequency_fused_output], dim=1)
        # fused: [B, 2 * c2, H/2, W/2]

        out = self.conv5(fused)
        # out: [B, c2, H/2, W/2]

        return out


class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma):
        super(LoGFilter, self).__init__()

        # 7x7 convolution with stride 1 for feature reinforcement
        self.conv_init = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=7,
            stride=1,
            padding=3
        )

        # Create LoG kernel
        ax = torch.arange(
            -(kernel_size // 2),
            (kernel_size // 2) + 1,
            dtype=torch.float32
        )

        xx, yy = torch.meshgrid(ax, ax, indexing='ij')

        kernel = (
            (xx ** 2 + yy ** 2 - 2 * sigma ** 2)
            / (2 * math.pi * sigma ** 4)
            * torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        )

        # Normalize LoG kernel
        kernel = kernel - kernel.mean()

        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        # log_kernel: [1, 1, kernel_size, kernel_size]

        self.LoG = nn.Conv2d(
            out_c,
            out_c,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
            groups=out_c,
            bias=False
        )

        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.LoG.weight.requires_grad = False

        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        # x: [B, in_c, H, W]

        x = self.conv_init(x)
        # x: [B, out_c, H, W]

        LoG = self.LoG(x)

        LoG_edge = self.act(self.norm1(LoG))

        x = self.norm2(x + LoG_edge)

        return x


class LoGStem(nn.Module):
    def __init__(self, in_chans, stem_dim):
        super().__init__()

        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)

        # original size to 2x downsampling layer
        # [B, stem_dim/4, H, W] -> [B, stem_dim/2, H/2, W/2]
        self.Conv_D = nn.Sequential(
            nn.Conv2d(
                out_c14,
                out_c12,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_c14
            ),
            Conv(out_c12, out_c12, 3, 2, g=out_c12)
        )

        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0)

        # Gaussian enhancement
        self.gaussian = Gaussian(out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(out_c12)

        self.drfd = FSConv(out_c12, stem_dim, k=3)

    def forward(self, x):
        # x: [B, in_chans, H, W]

        x = self.LoG(x)
        # x: [B, stem_dim/4, H, W]

        x = self.Conv_D(x)
        # x: [B, stem_dim/2, H/2, W/2]

        x = self.norm(x + self.gaussian(x))
        # x: [B, stem_dim/2, H/2, W/2]

        x = self.drfd(x)
        # x: [B, stem_dim, H/4, W/4]

        return x