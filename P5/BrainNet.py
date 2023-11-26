class BrainNet(nn.Module):
    def __init__(self, SuperFinal, SubFinal, resnet):
        super(BrainNet, self).__init__()
        self.SuperFinal = SuperFinal
        self.SubFinal = SubFinal
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet(x)
        x1 = self.SuperFinal(x)
        x2 = self.SubFinal(x)
        return x1, x2