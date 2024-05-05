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
    X_train, Y_train = generate_training_set(train_path)

    # Create a network with two layers of 27 neurons each (input and output layers)
    generator = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=generator, requires_grad=True) 

    # Gradient Descent
    for _ in range(100):
      """
      Forward propagation
      """
      logits = (X_train @ W) # Output layer results before aplying activation function
      
      # Apply sotmax as activation function in the output layer, used as the semantic operation needed to interpret the results of the output
      counts = logits.exp() # Convert the logits into counts

      probabilities = counts / counts.sum(1, keepdim=True) # Get the softmax result

      """
      Optmization
      """
      # Compute loss
      size = Y_train.size(0) # 228146 bigrams count
      loss = -probabilities[torch.arange(size), Y_train].log().mean()

      print(loss.item())

      # Backward pass
      W.grad = None
      loss.backward()

      # Update
      W.data += -50 * W.grad


def generate_training_set(train_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
  names = read_names(train_path)
  letters = get_unique_letters_list(names)
  stoi = encode_characters(letters)

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

