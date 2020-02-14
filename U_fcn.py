import torch.nn as nn

class U_FCN(nn.Module):

    def __init__(self, n_class):
        super(U_FCN, self).__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=(3,5), stride=(2,4), padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256 * 2, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128 * 2, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64 * 2, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.deconv5  = nn.ConvTranspose2d(32 * 2, 32 * 2, kernel_size=(3, 5), stride=(2,4), padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(32 * 2, n_class, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        pool = nn.MaxPool2d(2, stride=2, return_indices = True)
        unpool = nn.MaxUnpool2d(2, stride=2)

        x_1 = self.bnd1(self.relu(self.conv1(x)))
        x, indice1 = pool(x_1)
        x_2 = self.bnd2(self.relu(self.conv2(x)))
        x, indice2 = pool(x_2)
        x_3 = self.bnd3(self.relu(self.conv3(x)))
        x, indice3 = pool(x_3)
        x_4 = self.bnd4(self.relu(self.conv4(x)))
        x, indice4 = pool(x_4)
        x_5 = self.bnd5(self.relu(self.conv5(x)))

        z = self.bn1(self.relu(self.deconv1(x_5)))
        z = torch.cat([unpool((z), indice4), x_4], dim=1)
        z = self.bn2(self.relu(self.deconv2(z)))
        z = torch.cat([unpool((z), indice3), x_3], dim=1)
        z = self.bn3(self.relu(self.deconv3(z)))
        z = torch.cat([unpool((z), indice2), x_2], dim=1)
        z = self.bn4(self.relu(self.deconv4(z)))
        z = torch.cat([unpool((z), indice1), x_1], dim=1)
        z = self.bn5(self.relu(self.deconv5(z)))

        score = self.classifier(z)

        return score  