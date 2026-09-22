import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from modelo import MinhaCNN


# Transformação usada no treino e no teste
transformacao_treino = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

transformacao_teste = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# Dataset de treino
dataset = datasets.CIFAR10(
    root="./dados",
    train=True,
    download=False,
    transform=transformacao_treino
)

carregador = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# Dataset de teste
dataset_teste = datasets.CIFAR10(
    root="./dados",
    train=False,
    download=False,
    transform=transformacao_teste
)

carregador_teste = DataLoader(
    dataset_teste,
    batch_size=32,
    shuffle=False
)


# GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dispositivo:", device)


# Modelo
modelo = MinhaCNN().to(device)


# Otimizador
otimizador = torch.optim.SGD(
    modelo.parameters(),
    lr=0.01
)


# Função de perda
criterio = nn.CrossEntropyLoss()


# Treinamento
numero_epochs = 20

for epoch in range(numero_epochs):

    modelo.train()

    erro_total = 0.0

    for numero_batch, (imagens, classes) in enumerate(carregador):

        imagens = imagens.to(device)
        classes = classes.to(device)

        # Forward
        saida = modelo(imagens)

        # Calcula o erro
        erro = criterio(saida, classes)

        erro_total += erro.item()

        # Limpa gradientes antigos
        otimizador.zero_grad()

        # Backpropagation
        erro.backward()

        # Atualiza pesos
        otimizador.step()

        if numero_batch % 100 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {numero_batch} | "
                f"Erro: {erro.item():.4f}"
            )

    erro_medio = erro_total / len(carregador)

    print(
        f"Epoch {epoch + 1} finalizada | "
        f"Erro médio: {erro_medio:.4f}"
    )


# =========================
# AVALIAÇÃO
# =========================

modelo.eval()

corretas = 0
total = 0

with torch.no_grad():

    for imagens, classes in carregador_teste:

        imagens = imagens.to(device)
        classes = classes.to(device)

        saida = modelo(imagens)

        previsoes = torch.argmax(saida, dim=1)

        corretas += (previsoes == classes).sum().item()

        total += classes.size(0)


acuracia = corretas / total

print(f"Acurácia no teste: {acuracia * 100:.2f}%")


# =========================
# SALVAR MODELO
# =========================

torch.save(
    modelo.state_dict(),
    "modelo_cifar10.pth"
)

print("Modelo salvo!")


# =========================
# VISUALIZAR PREVISÕES
# =========================

imagens, classes = next(iter(carregador_teste))

imagens = imagens.to(device)
classes = classes.to(device)

with torch.no_grad():
    saidas = modelo(imagens)

previsoes = torch.argmax(saidas, dim=1)


fig, eixos = plt.subplots(
    2,
    4,
    figsize=(12, 6)
)


# Valores usados na normalização
media = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(3, 1, 1)

desvio = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(3, 1, 1)


for i, eixo in enumerate(eixos.flat):

    imagem = imagens[i].cpu()

    # Desnormaliza para visualizar corretamente
    imagem = imagem * desvio + media

    # [C, H, W] -> [H, W, C]
    imagem = imagem.permute(1, 2, 0)

    # Garante valores válidos para imagem
    imagem = imagem.clamp(0, 1)

    classe_real = dataset_teste.classes[
        classes[i].item()
    ]

    classe_prevista = dataset_teste.classes[
        previsoes[i].item()
    ]

    eixo.imshow(imagem)

    eixo.set_title(
        f"Real: {classe_real}\n"
        f"Previsto: {classe_prevista}"
    )

    eixo.axis("off")


plt.tight_layout()
plt.show()