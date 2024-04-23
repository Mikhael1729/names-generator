from textwrap import dedent
from typing import List, Dict, Tuple
import torch
import matplotlib.pyplot as plt
from bigram_model import BigramNameGenerator

DATASET_PATH = './datasets/names.txt'
START_SYMBOL = '.'
END_SYMBOL = '.' # The same as START_SYMBOL

def main():
  print(dedent(
    """
    Human: Give me a name!
    Machine: You are bestowed a name, and it is, "Mayah"
    ---
    """
  ))

  # e1_exploring_dataset()
  # e2_understanding_bigram()
  # e3_playing_with_tensors()
  # e4_understanding_bigram_using_2d_array()
  # e5_create_bigram_model()
  e6_use_bigram_model()

def e6_use_bigram_model():
  # Extract the features of the names
  generator = BigramNameGenerator(DATASET_PATH)

  # Generate a list containing 10 names
  names = generator.generate(10)
  print(names)

  # Visualize the features of the model
  generator.visualize()

def e5_create_bigram_model():
  """
  Convert the dictionary representation of the bigram into a 2D array
  """
  names = read_names(DATASET_PATH)
  letters = get_unique_letters_list(names)
  stoi = encode_characters(letters)
  itos = decode_characters(stoi)
  N = len(stoi) # Number of unique characters

  # Initialize bigram count matrix
  bigram_count = torch.zeros((N, N), dtype=torch.int32)

  # Count bigrams using the matrix
  for name in names:
    name_letters = [START_SYMBOL] + list(name) + [END_SYMBOL]

    for letter1, letter2 in zip(name_letters, name_letters[1:]):
      index_letter1 = stoi[letter1]
      index_letter2 = stoi[letter2]

      bigram_count[index_letter1, index_letter2] += 1

  # Visualize the results using a heatmap
  plt.imshow(bigram_count)

  # Visualize heatmap with data
  S = 16
  plt.figure(figsize=(S, S))
  plt.imshow(bigram_count, cmap='Blues')
  plt.axis('off')
  font_size = S / 2

  for i in range(N):
    for j in range(N):
      bigram = itos[i] + itos[j]
      plt.text(j, i, bigram, ha='center', va='bottom', color='gray', fontsize=font_size)
      plt.text(j, i, bigram_count[i, j].item(), ha='center', va='top', color='gray', fontsize=font_size)

  plt.show()

  print("10 generated words: \n")

  # Model in action
  random_index = 0
  randomness_generator = torch.Generator().manual_seed(2147483647)

  for i in range(10):
    generated_letters = []

    while True:
      start_letters_count = bigram_count[random_index].float() # Contains the count of times each letter starts a name
      start_letters_distribution =  start_letters_count / start_letters_count.sum()
      random_index = torch.multinomial(
        start_letters_distribution,
        num_samples=1, # We want only one letter
        replacement=True, # Sample index can be drawn again
        generator= randomness_generator # To replicate the results
      ).item()

      generated_letters.append(itos[random_index])

      if random_index == 0:
        break

    print('-', ''.join(generated_letters[0:len(generated_letters) - 1]))


def e4_understanding_bigram_using_2d_array():
  """
  Convert the dictionary representation of the bigram into a 2D array
  """
  names = read_names(DATASET_PATH)
  letters = get_unique_letters_list(names)
  stoi = encode_characters(letters)
  itos = decode_characters(stoi)
  N = len(stoi) # Number of unique characters

  # Initialize bigram count matrix
  bigram_count = torch.zeros((N, N), dtype=torch.int32)

  # Count bigrams using the matrix
  for name in names:
    name_letters = [START_SYMBOL] + list(name) + [END_SYMBOL]

    for letter1, letter2 in zip(name_letters, name_letters[1:]):
      index_letter1 = stoi[letter1]
      index_letter2 = stoi[letter2]

      bigram_count[index_letter1, index_letter2] += 1

  # Visualize the results using a heatmap
  plt.imshow(bigram_count)

  # Visualize heatmap with data
  S = 16
  plt.figure(figsize=(S, S))
  plt.imshow(bigram_count, cmap='Blues')
  plt.axis('off')
  font_size = S / 2

  for i in range(N):
    for j in range(N):
      bigram = itos[i] + itos[j]
      plt.text(j, i, bigram, ha='center', va='bottom', color='gray', fontsize=font_size)
      plt.text(j, i, bigram_count[i, j].item(), ha='center', va='top', color='gray', fontsize=font_size)

  plt.show()

def encode_characters(characters: List[str]) -> Dict[str, int]:
  stoi = {character: index + 1 for index, character in enumerate(characters)}
  stoi[START_SYMBOL] = 0

  return stoi

def decode_characters(encoded_characters: Dict[str, int]):
  return {index: character for character, index in encoded_characters.items()}

def e3_playing_with_tensors():
  A = torch.zeros((5, 5), dtype=torch.int32)

  for i in range(5):
    for j in range(5):
      if i == j:
        A[i, j] = 1

  print(A)

def e2_understanding_bigram():
  """
  (1) You learn the statistics about of which characters are likely to
      follow other characters. by counting how often any two character
      appear together

  (2) You can "allucinate" characters to demarcate meta information
      that you may also want to encode. i.e: like the likeness of a
      given character to appear at the begining/end of a word
  """
  names = read_names(DATASET_PATH)
  bigram_count = get_bigram_count(names)

  # Sort by count
  print(sorted(bigram_count.items(), key = lambda key_value: -key_value[1]))

def get_bigram_count(names: List[str]) -> Dict[Tuple[str, str], int]:
  bigram_count = {}

  for name in names:
    characters = ['<S>'] + list(name) + ['<E>'] # (e2.2)

    for char1, char2 in zip(characters, characters[1:]):
      bigram = (char1, char2)
      bigram_count[bigram] = bigram_count.get(bigram, 0) + 1

  return bigram_count

def e1_exploring_dataset():
  names = read_names(DATASET_PATH)

  print('\n'.join(names[:10]))
  print("---\n")

  print('Quantity of words: \t', len(names))
  print('Word with less characters: \t', min(len(n) for n in names))
  print('Word with the most characters: \t', max(len(n) for n in names))
  print('Unique characters count: \t', len(get_unique_letters_list(names)))

def get_unique_letters_list(names: List[str]) -> List[str]:
  return sorted(list(set(''.join(names))))


def read_names(file_path: str) -> List[str]:
  with open(file_path, 'r') as file:
    names = file.read().splitlines()

  return names

if __name__ == "__main__":
  main()