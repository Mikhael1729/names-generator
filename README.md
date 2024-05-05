# Name Generator

This repo is my exercise in learning the basics of generative AI through a simple model for generating name-like words.

## Bigram model (statistical approach)

You train the model using the bigram probability distribution of all the letters in the dataset. Generate a new word given a the starting letter.

### Training

The following is the algorithm to generate the probability distribution that is going to be used in the model for predictions

1. Let $L$ be the list of all letters including $.$ as the start/end symbol (an array of lenght $27$).

$$
L=\{a,b,c,...,z\}
$$

2. Let $F$ be a $27\times27$ matrix The bigram frecuency of letters:

   - Frecuency of a letter appearing at the start of a name
   - Frecuency of a letter appearing at end start of a name
   - Frecuency of a letter appearing after another letter

3. Compute the probability distribution $P$ of $F$. This will yield a $27 \times 27$ vector representing $P$.

### Prediction

4. Let $i$ be the index of the first letter you want to use (rangin from $0$ to $26$)
5. Given $P_i$ draw an index $i=j$ of $L$ acording to the probability distribution.
6. Get $L_i$
7. Collect $L_i$ in a list $W$
8. Is it $L_i = .$? Go to 5, else, finish

### Conceptual understanding

$$
\begin{align}
&\text{model.train}(P)\\
&\text{model.generate}(i)
\end{align}
$$

### The performance of the model

I used the negative log likelihood to evaluate how well the model currently is perforforming.

$$
-\text{log}\left(L(\theta \mid X \right)) = - \sum_{i=1}^{n}log\left(P(x_i \mid \theta \right))
$$

| Symbol               | Code     |
| -------------------- | -------- |
| $\theta$             | $P$      |
| $P(x_i \mid \theta)$ | $P_{ij}$ |

The lower this value is the better the model is, because it is giving high probabilities to the actual next characters in all the bigrams in the dataset, meaning the model is suer in predicting an specific name

## Bigram model (neural network approach)

| variable | shape      | name                | definiton                                                               |
| -------- | ---------- | ------------------- | ----------------------------------------------------------------------- |
| `X`      | `(5, )`    | encoded letters     | Integer values representing the letters of a name                       |
| `Y`      | `(5, )`    | encoded next letter | Integer values representing the next letters for `X`                    |
| `O`      | `(5, 27)`  | one-hot encoding    | The encoding of each sample in `X`                                      |
| `W`      | `(27, 27)` | weights             | Weights connecting the 27 input neurons with the 27 in the hidden layer |

```text
Forward pass:

logits = O * W { (5, 27) * (27, 27) -> (5, 27) } 

# Apply softmax
counts = logits.exp() // fake (counts sort of)
probs = counts / counts.sum(1, keepdims=True)

probs is a (5, 27). Which contains the probability distributions for the next letter on each of the 5 samples

5 letters
5 distribution probabilities
```
