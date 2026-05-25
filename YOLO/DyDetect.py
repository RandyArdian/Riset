class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention. HSigmoid arguments in default act_cfg follow official code, not
    paper. https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py.
    """

    def __init__(
        self, in_channels, norm_type="GN", zero_init_offset=True, act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0)
    ):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        if norm_type == "GN":
            norm_dict = dict(type="GN", num_groups=16, requires_grad=True)
        elif norm_type == "BN":
            norm_dict = dict(type="BN", requires_grad=True)

        self.spatial_conv_high = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)

        # self.spatial_conv_mid = DyDCNv2(in_channels, in_channels)

        # self.spatial_conv_mid = CBAM(in_channels)

        self.spatial_conv_mid = Conv(in_channels, in_channels, 3, 1)

        self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2)

        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
        )
        self.task_attn_module = DyReLUB(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # print(f'level: {level}')
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, : self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()

            # mid_feat = self.spatial_conv_mid(x[level], offset, mask)

            mid_feat = self.spatial_conv_mid(x[level])

            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                # print(f'if level > 0')
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # print(f'level < len(x) - 1')
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs


class DyDetect(Detect):
    def __init__(self, nc=10, ch=()):  # detection layer
        super().__init__(nc, ch)
        self.dyhead = nn.Sequential(*[DyHeadBlock(ch[0]) for _ in range(2)])
        self.cv2 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, self.nc, 1)) for x in ch)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for layer in self.dyhead:
            x = layer(x)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
