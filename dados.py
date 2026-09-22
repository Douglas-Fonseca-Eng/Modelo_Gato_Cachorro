import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transformacao = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./dados",
    train=True,
    download=True,
    transform=transformacao
)

print("Quantidade de imagens:", len(dataset))

# Pegando uma imagem individual
imagem, classe = dataset[0]

print("Formato da imagem:", imagem.shape)
print("Classe:", classe)
print("Nome da classe:", dataset.classes[classe])

# Preparando o carregador
carregador = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Visualizando a imagem individual
imagem_visual = imagem.permute(1, 2, 0)

plt.imshow(imagem_visual)
plt.show()
imagens, classes = next(iter(carregador))

print("Formato do batch de imagens:", imagens.shape)
print("Formato das classes:", classes.shape)
print(classes)