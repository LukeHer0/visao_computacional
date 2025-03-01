import cv2
import matplotlib.pyplot as plt
import numpy as np


def gaussian_pyramid(image, levels):
  """
    Cria uma pirâmide Gaussiana de uma imagem.

    Args:
        image: A imagem de entrada (numpy array).
        levels: O número de níveis na pirâmide.

    Returns:
        Uma lista de imagens, onde cada imagem é um nível da pirâmide.
    """
  pyramid = [image]
  for i in range(levels):
    image = cv2.pyrDown(image)  # Reduz a resolução da imagem
    pyramid.append(image)
  return pyramid


def display_pyramid(pyramid):
  """
    Exibe a pirâmide Gaussiana usando matplotlib.

    Args:
        pyramid: Uma lista de imagens representando a pirâmide.
    """
  num_levels = len(pyramid)
  plt.figure(figsize=(15, 5))  # Ajusta o tamanho da figura

  for i in range(num_levels):
    plt.subplot(1, num_levels, i + 1)
    plt.imshow(cv2.cvtColor(pyramid[i], cv2.COLOR_BGR2RGB))  # Converte para RGB
    plt.title(f"Level {i}")
    plt.axis("off")  # Desativa os eixos

  plt.tight_layout()  # Ajusta o layout para evitar sobreposição
  plt.show()


if __name__ == "__main__":
  # Carrega uma imagem usando OpenCV
  image = cv2.imread("img\polvo.jpg")  # Substitua pelo caminho da sua imagem

  if image is None:
    print("Erro: Não foi possível carregar a imagem.")
  else:
    # Define o número de níveis da pirâmide
    num_levels = 4

    # Cria a pirâmide Gaussiana
    pyramid = gaussian_pyramid(image, num_levels)

    # Exibe a pirâmide
    display_pyramid(pyramid)