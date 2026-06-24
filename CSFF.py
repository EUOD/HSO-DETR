class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
    
class SFC_G2(nn.Module):
    def __init__(self, inc):
        super(SFC_G2, self).__init__()
        hidc = inc[0]
        
        self.groups = 2
        self.conv_8 = Conv(inc[0], hidc, 3)
        self.conv_32 = Conv(inc[1], hidc, 3)

        self.conv_offset = nn.Sequential(
            Conv(hidc * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.keras_init_weight()
        self.conv_offset[1].weight.data.zero_()
        
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                
    def forward(self, x):
        cp, sp = x
        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征  1 / 8  256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True) # 上采样
        # x_8
        cp = self.conv_8(cp)

        conv_results = self.conv_offset(torch.cat([cp, sp], 1))

        sp = sp.reshape(n*self.groups,-1,out_h,out_w)
        cp = cp.reshape(n*self.groups,-1,out_h,out_w)

        offset_l = conv_results[:, 0:self.groups*2, :, :].reshape(n*self.groups,-1,out_h,out_w)
        offset_h = conv_results[:, self.groups*2:self.groups*4, :, :].reshape(n*self.groups,-1,out_h,out_w)


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n*self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l , align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h , align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups*4:, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp