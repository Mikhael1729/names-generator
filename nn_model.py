import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch.nn.functional as F
import torch

START_SYMBOL = '.'
END_SYMBOL = '.' # The same as START_SYMBOL

class BigramNNModel:
  def __init__(self, train_path: str):
    """
    Obtain training data
    """
    names, _, stoi, itos = load_raw_training_data(train_path)
    X_train, Y_train = generate_training_set(names, stoi)

    self.itos = itos
    self.num_classes = len(stoi)

    self.train(X_train, Y_train)

  def train(self, X_train: torch.Tensor, Y_train: torch.Tensor, iterations=100, step_size=50):
    # Create a network with two layers of 27 neurons each (input and output layers)
    generator = torch.Generator().manual_seed(2147483647)
    W = torch.randn((self.num_classes, self.num_classes), generator=generator, requires_grad=True) 

    # Gradient Descent
    for _ in range(iterations):
      """
      Forward propagation
      """
      logits = (X_train @ W) # Output layer results before aplying activation function
      
      # Apply sotmax as activation function in the output layer, used as the semantic operation needed to interpret the results of the output
      counts = logits.exp() # Convert the logits into counts

      probabilities = counts / counts.sum(1, keepdim=True) # Get the softmax result

      """
      Optimization
      """
      # Compute loss
      size = Y_train.size(0) # 228146 bigrams count
      loss = -probabilities[torch.arange(size), Y_train].log().mean()

      print(loss.item())

      # Backward pass
      W.grad = None
      loss.backward()

      # Update
      W.data += -step_size * W.grad

    self.W = W

  def generate(self, quantity):
    generator = torch.Generator().manual_seed(2147483647)
    generated_names = []
    random_index = 0

    for _ in range(quantity):
      generated_letters = []

      while True:
        # Forward
        x_hot = self.encode(torch.tensor([random_index]))
        logits = (x_hot @ self.W)
        counts = logits.exp()
        probabilities = counts / counts.sum(1, keepdim=True) # Get the softmax result

        random_index = torch.multinomial(
          probabilities,
          num_samples=1, # We want only one letter
          replacement=True, # Sample index can be drawn again
          generator=generator # To replicate the results
        ).item()

        generated_letters.append(self.itos[random_index])

        if random_index == 0:
          break

      generated_names.append(''.join(generated_letters[0:len(generated_letters) - 1]))

    return generated_names

  def encode(self, index: torch.Tensor):
    X_hot = F.one_hot(index, num_classes=self.num_classes).float()
    return X_hot 


def generate_training_set(names: List[str], stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
  X = [] # Sample inputs
  Y = [] # Expected letters (outputs)

  for name in names:
    name_letters = [START_SYMBOL] + list(name) + [END_SYMBOL]

    for letter1, letter2 in zip(name_letters, name_letters[1:]):
      index_letter1 = stoi[letter1]
      index_letter2 = stoi[letter2]

      X.append(index_letter1)
      Y.append(index_letter2)

  X = torch.tensor(X)
  Y = torch.tensor(Y)

  X_hot = F.one_hot(X, num_classes=len(stoi)).float()

  return X_hot, Y


def load_raw_training_data(training_set_path: str) -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str]]:
  names = read_names(training_set_path)
  letters = get_unique_letters_list(names)
  stoi = encode_characters(letters)
  itos = decode_characters(stoi)

  return names, letters, stoi, itos


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

