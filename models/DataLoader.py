import numpy as np
import jax.numpy as jnp
import jax.random as jr

class DataLoader:
    def __init__(self, filename, key):
        """
            Train Test Split always remains to same to make test losses comparable across runs.

            Only batch sampling order and weight initialization for the models are varied.
        """
        self.batch_sampling_key = key

        self.raw_data = np.load(filename)
        self.raw_data = jr.permutation(key=jr.key(23456), x=self.raw_data)
        self.data = self.transform(self.raw_data.copy())

        N = len(self.data)
        self.train_split = 4 * N // 5
        self.validation_split = 9 * N // 10

        self.train_data = self.data[:self.train_split]
        self.validation_data = self.data[self.train_split:self.validation_split]
        self.test_data = self.data[self.validation_split:]

        
    def transform(self, data):
        return jnp.log10(1 + data)

    def sample_batch(self, batch_size):
        self.batch_sampling_key, subkey = jr.split(self.batch_sampling_key)
        idx = jr.choice(key=subkey, a=self.train_split, shape=(batch_size,))
        return self.train_data[idx]

    def __len__(self):
        return len(self.data)