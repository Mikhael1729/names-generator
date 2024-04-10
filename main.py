from textwrap import dedent
from typing import List, Dict, Tuple

DATASET_PATH = './datasets/names.txt'

def main():
  print(dedent(
    """
    Human: Give me a name!
    Machine: You are bestowed a name, and it is, "Mayah"
    ---
    """
  ))

  # e1_exploring_dataset()
  e2_understanding_bigram()


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


def read_names(file_path: str) -> List[str]:
  with open(file_path, 'r') as file:
    names = file.read().splitlines()

  return names

if __name__ == "__main__":
  main()