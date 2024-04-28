import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch

START_SYMBOL = '.'
END_SYMBOL = '.' # The same as START_SYMBOL

class BigramNameGenerator:
  def __init__(self, train_path: str):
    names = read_names(train_path)
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

    self.bigram_count = bigram_count
    self.N = N
    self.itos = itos
    self.stoi = stoi
    self.names = names
    self.bigrams_probability_distribution = None

  def generate(self, quantity: int) -> List[str]:
    generated_names = []
    random_index = 0
    randomness_generator = torch.Generator().manual_seed(2147483647)

    all_letters_distribution = self.bigram_count.float()
    all_letters_distribution /= self.bigram_count.sum(1, keepdim=True)
    # Two thing about the last expression:
    # 1. The reason I used an inplace operation instead of doing p = p / p.sum(...) is for eficiency,
    #    this way it avoids creating a new complete tensor an just updates the existing one
    # 2. The broadcasting: Division is applied to a (27,27) / (27, 1)
    # 3. About the parameters: 
    #    1            : indicates the sum applies only to dimension 1 (the rows)
    #    keepdim=True : It tells the function to avoid squeezing dimensions if they become of size 1, in this case the rows

    self.bigrams_probability_distribution = all_letters_distribution

    for _ in range(quantity):
      generated_letters = []

      while True:
        letters_distribution = all_letters_distribution[random_index]
        random_index = torch.multinomial(
          letters_distribution,
          num_samples=1, # We want only one letter
          replacement=True, # Sample index can be drawn again
          generator=randomness_generator # To replicate the results
        ).item()

        generated_letters.append(self.itos[random_index])

        if random_index == 0:
          break

      generated_names.append(''.join(generated_letters[0:len(generated_letters) - 1]))

    return generated_names

  def get_negative_log_likelihood(self, normalized=True):
    return -1 * self.get_log_likelihood(normalized)

  def get_log_likelihood(self, normalized) -> float:
    log_likelihood = 0.0
    n = 0

    for name in self.names:
      name_letters = [START_SYMBOL] + list(name) + [END_SYMBOL]

      for letter1, letter2 in zip(name_letters, name_letters[1:]):
        index_letter1 = self.stoi[letter1]
        index_letter2 = self.stoi[letter2]

        # The probability of occuring any two secuence of words
        probability = self.bigrams_probability_distribution[index_letter1, index_letter2]

        log_probability = torch.log(probability)
        log_likelihood += log_probability

        n += 1

    return log_likelihood if not normalized else normalized / n

  def visualize(self):
    # Visualize the results using a heatmap
    plt.imshow(self.bigram_count)

    # Visualize heatmap with data
    S = 16
    plt.figure(figsize=(S, S))
    plt.imshow(self.bigram_count, cmap='Blues')
    plt.axis('off')
    font_size = S / 2

    for i in range(self.N):
      for j in range(self.N):
        bigram = self.itos[i] + self.itos[j]
        plt.text(j, i, bigram, ha='center', va='bottom', color='gray', fontsize=font_size)
        plt.text(j, i, self.bigram_count[i, j].item(), ha='center', va='top', color='gray', fontsize=font_size)

    plt.show()

def decode_characters(encoded_characters: Dict[str, int]):
  return {index: character for character, index in encoded_characters.items()}

def encode_characters(characters: List[str]) -> Dict[str, int]:
  stoi = {character: index + 1 for index, character in enumerate(characters)}
  stoi[START_SYMBOL] = 0

  return stoi

def get_unique_letters_list(names: List[str]) -> List[str]:
  return sorted(list(set(''.join(names))))

def read_names(file_path: str) -> List[str]:
  with open(file_path, 'r') as file:
    names = file.read().splitlines()

  return names
