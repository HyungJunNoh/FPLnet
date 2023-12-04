import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
  layers = []
  layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        bias=bias)]
  torch.nn.init.kaiming_uniform_(layers[0].weight)
  layers += [nn.BatchNorm2d(num_features=out_channels)]
  layers += [nn.ReLU()]
  cbr = nn.Sequential(*layers)

  return cbr

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()

    self.enc1_1 = CBR2d(in_channels=3, out_channels=16) 
    self.enc1_2 = CBR2d(in_channels=16, out_channels=16)

    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.enc2_1 = CBR2d(in_channels=16, out_channels=32)
    self.enc2_2 = CBR2d(in_channels=32, out_channels=32)

    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.enc3_1 = CBR2d(in_channels=32, out_channels=64)
   
    self.dec3_1 = CBR2d(in_channels=64, out_channels=32)

    self.unpool2_upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
    self.unpool2_conv = CBR2d(in_channels=32, out_channels=32)
    
    
    self.dec2_2 = CBR2d(in_channels=2 * 32, out_channels=32)
    self.dec2_1 = CBR2d(in_channels=32, out_channels=16)

    self.unpool1_upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
    self.unpool1_conv = CBR2d(in_channels=16, out_channels=16)
    
    self.dec1_2 = CBR2d(in_channels=2 * 16, out_channels=16)
    self.dec1_1 = CBR2d(in_channels=16, out_channels=16)

    self.fc = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

  def forward(self, x):
    enc1_1 = self.enc1_1(x)
    enc1_2 = self.enc1_2(enc1_1)
    pool1 = self.pool1(enc1_2)

    enc2_1 = self.enc2_1(pool1)
    enc2_2 = self.enc2_2(enc2_1)
    pool2 = self.pool2(enc2_2)

    enc3_1 = self.enc3_1(pool2)

    dec3_1 = self.dec3_1(enc3_1)

    unpool2 = self.unpool2_upsample(dec3_1)
    unpool2 = self.unpool2_conv(unpool2)
    cat2 = torch.cat((unpool2, enc2_2), dim=1)
    dec2_2 = self.dec2_2(cat2)
    dec2_1 = self.dec2_1(dec2_2)

    unpool1 = self.unpool1_upsample(dec2_1)
    unpool1 = self.unpool1_conv(unpool1)
    cat1 = torch.cat((unpool1, enc1_2), dim=1)
    dec1_2 = self.dec1_2(cat1)
    dec1_1 = self.dec1_1(dec1_2)

    x = self.fc(dec1_1)

    return x