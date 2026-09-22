<div align="center">

# 🧠 CNN com PyTorch no CIFAR-10

### Projeto introdutório de Deep Learning e Visão Computacional

Construção de uma **Rede Neural Convolucional (CNN) do zero com PyTorch**, passando pelas principais etapas de um pipeline de aprendizado profundo: preparação dos dados, treinamento em GPU, backpropagation, otimização, avaliação e inferência.

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-blue?style=for-the-badge)

<br>

**Melhor acurácia obtida no conjunto de teste: `71,03%`**

</div>

---

## 📌 Sobre o projeto

Este projeto foi desenvolvido com o objetivo de compreender, na prática, os principais processos envolvidos na criação e treinamento de um modelo de **Deep Learning para classificação de imagens**.

O foco não é apenas obter uma boa acurácia, mas entender cada etapa do processo:

```text
Imagem
   ↓
Dataset
   ↓
Pré-processamento
   ↓
DataLoader
   ↓
CNN
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Atualização dos pesos
   ↓
Avaliação
   ↓
Inferência
```

O modelo foi implementado utilizando **PyTorch** e treinado com aceleração por **GPU NVIDIA utilizando CUDA**.

---

## 🎯 Objetivos de aprendizagem

Durante o desenvolvimento foram estudados e implementados conceitos como:

- Tensores
- Dataset e DataLoader
- Batch e Epoch
- Redes Neurais Convolucionais
- Filtros convolucionais
- Feature Maps
- ReLU
- MaxPooling
- Batch Normalization
- Dropout
- Normalização de imagens
- Data Augmentation
- Cross Entropy Loss
- Gradientes
- Backpropagation
- Otimizadores
- Treinamento utilizando CUDA
- Avaliação em conjunto de teste
- Salvamento e carregamento dos pesos
- Inferência em imagens externas

---

# 🗂️ Dataset

Foi utilizado o **CIFAR-10**, um dataset amplamente utilizado no estudo de visão computacional.

Ele possui:

| Conjunto | Quantidade |
|---|---:|
| Treinamento | 50.000 imagens |
| Teste | 10.000 imagens |
| Total | 60.000 imagens |

Cada imagem possui:

```text
32 × 32 pixels
3 canais RGB
```

As imagens pertencem a **10 classes**:

```text
0 → airplane
1 → automobile
2 → bird
3 → cat
4 → deer
5 → dog
6 → frog
7 → horse
8 → ship
9 → truck
```

---

# 🧠 Arquitetura da CNN

A arquitetura desenvolvida utiliza duas camadas convolucionais.

```text
Imagem RGB
[3 × 32 × 32]

        ↓

Conv2D
3 → 16 filtros
Kernel 3×3

        ↓

Batch Normalization

        ↓

ReLU

        ↓

MaxPool 2×2

        ↓

[16 × 15 × 15]

        ↓

Conv2D
16 → 32 filtros
Kernel 3×3

        ↓

Batch Normalization

        ↓

ReLU

        ↓

MaxPool 2×2

        ↓

[32 × 6 × 6]

        ↓

Flatten

        ↓

1152 valores

        ↓

Camada Linear

        ↓

10 logits

        ↓

Classe prevista
```

---

## 🔎 Como a CNN interpreta uma imagem

Uma CNN não recebe conceitos como:

```text
"isso é um gato"
```

Ela recebe números correspondentes aos pixels.

Por exemplo:

```text
Imagem
↓
Tensor [3, 32, 32]
↓
Convoluções
↓
Bordas / texturas / padrões
↓
Características mais complexas
↓
Classificação
```

As convoluções aprendem automaticamente filtros durante o treinamento.

---

# ⚙️ Pipeline de treinamento

O modelo é treinado em batches de:

```python
batch_size = 32
```

Para cada batch:

```text
32 imagens
      ↓
Forward Pass
      ↓
Previsões
      ↓
CrossEntropyLoss
      ↓
backward()
      ↓
Gradientes
      ↓
optimizer.step()
      ↓
Pesos atualizados
```

O processo é repetido por várias epochs.

---

# 🧮 Função de perda

Foi utilizada:

```python
nn.CrossEntropyLoss()
```

A função compara os logits produzidos pela rede com a classe correta.

Exemplo:

```text
Saída da rede

[-0.2, 0.4, 0.1, 1.8, ...]

Classe correta

3

      ↓

CrossEntropyLoss

      ↓

Erro
```

---

# 🔁 Backpropagation

Após calcular o erro:

```python
erro.backward()
```

o PyTorch calcula os gradientes dos parâmetros da rede.

Depois:

```python
otimizador.step()
```

atualiza os pesos do modelo.

De forma simplificada:

```text
peso novo =
peso antigo - learning_rate × gradiente
```

---

# 🎛️ Otimizador

Foi utilizado inicialmente:

```python
torch.optim.SGD(
    modelo.parameters(),
    lr=0.01
)
```

Onde:

```text
SGD → algoritmo responsável pela atualização dos pesos

lr → learning rate
```

---

# 📊 Evolução dos experimentos

Durante o desenvolvimento, diferentes técnicas foram testadas.

| Experimento | Acurácia |
|---|---:|
| CNN inicial | 62,37% |
| + Normalização | 68,89% |
| + Data Augmentation | 69,70% |
| + Batch Normalization | **71,03%** |
| + Dropout 0.5 | 69,94% |
| + Dropout reduzido | 70,66% |

### Melhor resultado

```text
Acurácia no teste: 71,03%
```

O resultado foi obtido em **10.000 imagens que não participaram do treinamento**.

---

# 📈 O que os experimentos mostraram

Um dos principais aprendizados do projeto foi perceber que aumentar o desempenho de uma rede neural não depende apenas de aumentar o número de epochs.

Mudanças no pré-processamento e na arquitetura tiveram impactos significativos.

A normalização, por exemplo, aumentou a acurácia de:

```text
62,37%
   ↓
68,89%
```

A adição de **Batch Normalization** levou o modelo a:

```text
71,03%
```

Também foi possível observar que uma técnica não necessariamente melhora todos os modelos.

O uso de:

```python
Dropout(0.5)
```

reduziu a acurácia, mostrando a importância de **realizar experimentos e medir os resultados**, em vez de assumir que uma técnica sempre produzirá melhoria.

---

# 🖥️ Treinamento com GPU

O projeto utiliza automaticamente CUDA quando disponível:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

O modelo é movido para a GPU:

```python
modelo = MinhaCNN().to(device)
```

E os batches também:

```python
imagens = imagens.to(device)
classes = classes.to(device)
```

Ambiente utilizado durante o desenvolvimento:

```text
GPU: NVIDIA GeForce RTX 5060
PyTorch: 2.14.0 + CUDA
```

---

# 💾 Salvamento do modelo

Depois do treinamento, os pesos podem ser armazenados:

```python
torch.save(
    modelo.state_dict(),
    "modelo_cifar10.pth"
)
```

Isso permite reutilizar o modelo posteriormente sem realizar todo o treinamento novamente.

---

# 🔮 Inferência

O arquivo `teste.py` carrega novamente a arquitetura:

```python
modelo = MinhaCNN()
```

Depois carrega os pesos:

```python
modelo.load_state_dict(
    torch.load("modelo_cifar10.pth")
)
```

E coloca a rede em modo de avaliação:

```python
modelo.eval()
```

Assim é possível fornecer uma nova imagem e obter uma previsão:

```text
Imagem
↓
Pré-processamento
↓
CNN treinada
↓
10 logits
↓
argmax
↓
Classe prevista
```

---

# 📁 Estrutura do projeto

```text
Modelo_Gato_Cachorro/
│
├── modelo.py
│   └── Arquitetura da CNN
│
├── treino.py
│   └── Treinamento e avaliação
│
├── teste.py
│   └── Inferência utilizando modelo treinado
│
├── modelo_cifar10.pth
│   └── Pesos treinados
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

O dataset e o ambiente virtual não precisam ser armazenados no repositório:

```text
.venv/
dados/
__pycache__/
```

---

# 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/Douglas-Fonseca-Eng/Modelo_Gato_Cachorro.git
```

Entre na pasta:

```bash
cd Modelo_Gato_Cachorro
```

---

### 2. Crie um ambiente virtual

Windows:

```bash
python -m venv .venv
```

Ative:

```bash
.venv\Scripts\activate
```

---

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### 4. Treine o modelo

```bash
python treino.py
```

Durante o treinamento serão exibidos valores semelhantes a:

```text
Epoch 1 | Batch 0 | Erro: 2.31
Epoch 1 | Batch 100 | Erro: 2.27
...
Epoch 20 finalizada | Erro médio: ...
Acurácia no teste: ...
```

---

### 5. Execute uma inferência

```bash
python teste.py
```

Exemplo:

```text
Classe prevista: cat
```

---

# 🧪 Tecnologias utilizadas

<div align="center">

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem principal |
| PyTorch | Construção e treinamento da CNN |
| Torchvision | Dataset e transformações |
| CUDA | Aceleração utilizando GPU |
| Matplotlib | Visualização das imagens |
| Pillow | Manipulação de imagens externas |
| Git | Versionamento |
| GitHub | Hospedagem do projeto |

</div>

---

# 🛣️ Próximos passos

O projeto ainda pode evoluir com:

- [ ] Adicionar terceira camada convolucional
- [ ] Testar diferentes números de filtros
- [ ] Comparar SGD com Adam
- [ ] Utilizar Learning Rate Scheduler
- [ ] Criar matriz de confusão
- [ ] Calcular precisão, recall e F1-score
- [ ] Visualizar feature maps da CNN
- [ ] Visualizar filtros aprendidos
- [ ] Utilizar Transfer Learning
- [ ] Treinar com imagens de maior resolução
- [ ] Exportar o modelo para ONNX
- [ ] Integrar o modelo em aplicativo Android com Kotlin
- [ ] Executar inferência diretamente no celular

---

# 📱 Possível integração com Android

Uma evolução futura do projeto é exportar o modelo treinado:

```text
PyTorch
   ↓
Modelo treinado
   ↓
ONNX / formato para inferência móvel
   ↓
Android
   ↓
Kotlin
   ↓
Câmera
   ↓
Modelo
   ↓
Classificação
```

Isso permitiria utilizar o modelo treinado em uma aplicação mobile.

---

# 🎓 Contexto

Este projeto faz parte dos meus estudos em:

```text
Machine Learning
Deep Learning
Visão Computacional
PyTorch
Redes Neurais Convolucionais
```

O objetivo é construir uma base sólida antes de avançar para arquiteturas e aplicações mais complexas.

---

<div align="center">

## 👨‍💻 Autor

**Francisco Douglas**

Estudante de Engenharia da Computação  
Interesse em **Inteligência Artificial, Deep Learning e Visão Computacional**

<br>

[![GitHub](https://img.shields.io/badge/GitHub-Douglas--Fonseca--Eng-181717?style=for-the-badge&logo=github)](https://github.com/Douglas-Fonseca-Eng)

<br>

### ⭐ Projeto desenvolvido para aprendizado e experimentação em Deep Learning.

</div>