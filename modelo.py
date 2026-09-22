import torch
import torch.nn as nn


class MinhaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        self.fc = nn.Linear(32 * 6 * 6, 10)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)


        return x


if __name__ == "__main__":
    modelo = MinhaCNN()

    imagem_teste = torch.randn(1, 3, 32, 32)

    saida = modelo(imagem_teste)

    print("Entrada:", imagem_teste.shape)
    print("Saída:", saida.shape)
    print(saida)