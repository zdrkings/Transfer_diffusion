import torch


def sinusoidal_embedding(n, dim):
    # Returns the standard positional embedding,在这里是对每一个t进行编码，详细的过程记录在ipad上面
    embedding = torch.zeros(n, dim)
    wk = torch.tensor([1 / 10_000 ** (2 * j / dim) for j in range(dim)])
    wk = wk.reshape((1, dim))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding