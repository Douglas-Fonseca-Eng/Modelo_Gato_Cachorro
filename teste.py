import torch
from PIL import Image
from torchvision import transforms

from modelo import MinhaCNN


# =========================
# DISPOSITIVO
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# =========================
# CARREGAR MODELO
# =========================

modelo = MinhaCNN().to(device)

modelo.load_state_dict(
    torch.load(
        "modelo_cifar10.pth",
        map_location=device
    )
)

modelo.eval()

print("Modelo carregado com sucesso!")


# =========================
# CLASSES DO CIFAR-10
# =========================

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# =========================
# TRANSFORMAÇÃO
# =========================

transformacao = transforms.Compose([
    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# =========================
# CARREGAR IMAGEM
# =========================

imagem = Image.open(
    "dog.jpg"
).convert("RGB")


# Aplica as transformações
imagem = transformacao(imagem)


# [3, 32, 32]
# vira
# [1, 3, 32, 32]

imagem = imagem.unsqueeze(0)


# Manda para GPU
imagem = imagem.to(device)


# =========================
# INFERÊNCIA
# =========================

with torch.no_grad():

    saida = modelo(imagem)


# Pega a posição com maior valor
previsao = torch.argmax(
    saida,
    dim=1
).item()


# =========================
# RESULTADO
# =========================

print(
    "Classe prevista:",
    classes[previsao]
)