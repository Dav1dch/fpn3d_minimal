
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet3d import resnet18, resnet10


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, first_batch):
        super(FPN, self).__init__()
        # self.first_run = True
        self.in_planes = 64

        self.back_bone = resnet18()
        # print("Back bone:\n", self.back_bone)

        # Bottom-up layers
        # self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)

        # Top layer
        # nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer = None

        # Smooth layers
        # self.smooth1 = None  # nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = None  # nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = None  # nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = None
        # nn.Conv3d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = None
        # nn.Conv3d(64, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = None

        # Addendum layers to reduce channels before sum
        self.sumlayer1 = None
        self.sumlayer2 = None
        self.sumlayer3 = None

        # Semantic branch
        self.conv2_3d_p5 = None
        self.conv2_3d_p4 = None
        self.conv2_3d_p3 = None
        self.conv2_3d_p2 = None

        self.resumlayer1 = None
        self.relatlayer1 = None
        self.resumlayer2 = None
        self.relatlayer2 = None
        self.resumlayer3 = None
        self.relatlayer3 = None
        self.downlayer = None

        self.iam_joking(first_batch, False)

        self.semantic_branch_2d = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        # 3 is the number of samples taken from the time series
        self.conv4out = nn.Conv2d(
            64, 3, kernel_size=3, stride=1, padding=1)
        self.conv5out = nn.Conv2d(
            3, 128, kernel_size=3, stride=1, padding=1)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)
        self.pool = GeM()

    def iam_joking(self, x, use_cuda):
        low_level_features = self.back_bone(x)
        c1 = low_level_features[0]
        c2 = low_level_features[1]
        c3 = low_level_features[2]
        c4 = low_level_features[3]
        c5 = low_level_features[4]

        # Top layer
        self.toplayer = nn.Conv3d(c5.size()[1], c5.size(
        )[1], kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv3d(c4.size()[1], c4.size()[
                                   1], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(c3.size()[1], c3.size()[
                                   1], kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(c2.size()[1], c2.size()[
                                   1], kernel_size=1, stride=1, padding=0)

        # Addendum layers to reduce channels
        self.sumlayer1 = nn.Conv3d(c5.size()[1], c4.size(
        )[1], kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.sumlayer2 = nn.Conv3d(c4.size()[1], c3.size()[
                                   1], kernel_size=1, stride=1, padding=0)
        self.sumlayer3 = nn.Conv3d(c3.size()[1], c2.size()[
                                   1], kernel_size=1, stride=1, padding=0)

        if use_cuda:
            self.toplayer = self.toplayer.cuda()
            self.latlayer1 = self.latlayer1.cuda()
            self.latlayer2 = self.latlayer2.cuda()
            self.latlayer3 = self.latlayer3.cuda()
            self.sumlayer1 = self.sumlayer1.cuda()
            self.sumlayer2 = self.sumlayer2.cuda()
            self.sumlayer3 = self.sumlayer3.cuda()

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(self.sumlayer1(p5), self.latlayer1(c4))
        p3 = self._upsample_add(self.sumlayer2(p4), self.latlayer2(c3))
        p2 = self._upsample_add(self.sumlayer3(p3), self.latlayer3(c2))

        self.downlayer = nn.Conv3d(p2.size()[1], p2.size()[1], kernel_size=[
                                   2, 3, 3], stride=1, padding=[0, 1, 1])

        self.resumlayer1 = nn.Conv3d(p3.size()[1], p3.size()[1], kernel_size=[
                                     1, 3, 3], padding=[0, 1, 1])
        self.relatlayer1 = nn.Conv3d(p2.size()[1], p3.size()[1], kernel_size=[
                                     1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.resumlayer2 = nn.Conv3d(p4.size()[1], p4.size()[1], kernel_size=[
                                     1, 3, 3], padding=[0, 1, 1])
        self.relatlayer2 = nn.Conv3d(p3.size()[1], p4.size()[1], kernel_size=[
                                     1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.resumlayer3 = nn.Conv3d(p5.size()[1], p5.size()[1], kernel_size=[
                                     1, 3, 3], padding=[0, 1, 1])
        self.relatlayer3 = nn.Conv3d(p4.size()[1], p5.size()[1], kernel_size=[
                                     1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        if use_cuda:
            self.downlayer = self.downlayer.cuda()
            self.relatlayer1 = self.relatlayer1.cuda()
            self.resumlayer1 = self.resumlayer1.cuda()
            self.relatlayer2 = self.relatlayer2.cuda()
            self.resumlayer2 = self.resumlayer2.cuda()
            self.relatlayer3 = self.relatlayer3.cuda()
            self.resumlayer3 = self.resumlayer3.cuda()

        # Smooth layers
        # self.smooth1 = nn.Conv3d(p4.size()[1], 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv3d(p3.size()[1], 256, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv3d(p2.size()[1], 256, kernel_size=3, stride=1, padding=1)
        # if use_cuda:
        #     self.smooth1 = self.smooth1.cuda()
        #     self.smooth2 = self.smooth2.cuda()
        #     self.smooth3 = self.smooth3.cuda()
        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)

        # calculate the sizes so that dimension c becomes 1
        self.conv2_3d_p5 = nn.Conv3d(p5.size()[1], 256, kernel_size=(
            p5.size()[2] + 2, 3, 3), stride=1, padding=1)
        self.conv2_3d_p4 = nn.Conv3d(p4.size()[1], 256, kernel_size=(
            p4.size()[2] + 2, 3, 3), stride=1, padding=1)
        self.conv2_3d_p3 = nn.Conv3d(p3.size()[1], 128, kernel_size=(
            p3.size()[2] + 2, 3, 3), stride=1, padding=1)
        self.conv2_3d_p2 = nn.Conv3d(p2.size()[1], 128, kernel_size=(
            p2.size()[2] + 2, 3, 3), stride=1, padding=1)

    def _upsample3d(self, x, d, h, w):
        return F.interpolate(x, size=(d, h, w), mode='trilinear', align_corners=True)

    def _upsample2d(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _make_layer(self, Bottleneck, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, D, H, W = y.size()
        return F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=True) + y

    def forward(self, x):
        # x = x['images']
        # Bottom-up using backbone
        low_level_features = self.back_bone(x)
        c1 = low_level_features[0]
        c2 = low_level_features[1]
        c3 = low_level_features[2]
        c4 = low_level_features[3]
        c5 = low_level_features[4]

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(
            torch.relu(self.sumlayer1(p5)), torch.relu(self.latlayer1(c4)))  # p5 interpolation to the size of c4
        p3 = self._upsample_add(
            torch.relu(self.sumlayer2(p4)), torch.relu(self.latlayer2(c3)))
        p2 = self._upsample_add(
            torch.relu(self.sumlayer3(p3)), torch.relu(self.latlayer3(c2)))

        r2 = self.downlayer(p2)
        r3 = torch.relu(self.resumlayer1(p3)) + \
            torch.relu(self.relatlayer1(r2))
        r4 = torch.relu(self.resumlayer2(p4)) + \
            torch.relu(self.relatlayer2(r3))
        r5 = torch.relu(self.resumlayer3(p5)) + \
            torch.relu(self.relatlayer3(r4))
        # r5 = torch.squeeze(r3, 2)
        out_mini = r5.squeeze(2)

        '''
        # Semantic
        _, _, _, h, w = p2.size()
        h = h // 2
        w = w // 2
        # 256->256
        s5 = self.conv2_3d_p5(p5)
        # squeeze only dim 2 to avoid to remove the batch dimension
        s5 = torch.squeeze(s5, 2)
        s5 = self._upsample2d(torch.relu(self.gn2(s5)), h, w)
        # 256->256 [32, 256, 24, 24]
        s5 = self._upsample2d(torch.relu(self.gn2(self.conv2_2d(s5))), h, w)
        # 256->128 [32, 128, 24, 24]
        s5 = self._upsample2d(torch.relu(
            self.gn1(self.semantic_branch_2d(s5))), h, w)

        # 256->256 p4:[32, 256, 4, 6, 6] -> s4:[32, 256, 1, 6, 6]
        s4 = self.conv2_3d_p4(p4)
        s4 = torch.squeeze(s4, 2)  # s4:[32, 256, 6, 6]
        s4 = self._upsample2d(torch.relu(self.gn2(s4)),
                              h, w)  # s4:[32, 256, 24, 24]
        # 256->128  s4:[32, 128, 24, 24]
        s4 = self._upsample2d(torch.relu(
            self.gn1(self.semantic_branch_2d(s4))), h, w)

        # 256->128
        s3 = self.conv2_3d_p3(p3)
        s3 = torch.squeeze(s3, 2)
        s3 = self._upsample2d(torch.relu(self.gn1(s3)), h, w)

        s2 = self.conv2_3d_p2(p2)
        s2 = torch.squeeze(s2, 2)
        s2 = self._upsample2d(torch.relu(self.gn1(s2)), h, w)

        out = self._upsample2d(self.conv3(s2 + s3 + s4 + s5), 2 * h, 2 * w)
        # introducing MSELoss on NDVI signal
        # for Class Activation Interval

        out_cai = torch.sigmoid(self.conv4out(out))
        out_cls = self.conv5out(out_cai)  # for Classification
        # print("out_cai ", out_cai.shape)
        print("out_cls ", out_cls.shape)
        '''

        # out_dim = self.pool(out_cls)
        out_dim = self.pool(out_mini)
        # print(f"x_pool: {x.shape}")
        # x is (batch_size, 512, 1, 1) tensor

        out_dim = torch.flatten(out_dim, 1)
        print("out_dim: ", out_dim.shape)

        # return out_cai, out_cls
        return out_dim
