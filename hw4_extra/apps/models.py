import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBN(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv = ndl.nn.Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = bias, device = device, dtype = dtype)
        self.batch_norm = ndl.nn.BatchNorm2d(dim=out_channels, device=device, dtype=dtype)
        self.relu = ndl.nn.ReLU()
        
    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.relu(self.batch_norm(self.conv(x)))
        ### END YOUR SOLUTION

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        bias = True
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = ConvBN(3, 16, 7, 4, bias=bias, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, 3, 2, bias=bias, device=device, dtype=dtype)
        self.res = ndl.nn.Residual(
            ndl.nn.Sequential(
                ConvBN(32, 32, 3, 1, bias=bias, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, bias=bias, device=device, dtype=dtype)
            )
        )
        self.conv3 = ConvBN(32, 64, 3, 2, bias=bias, device=device, dtype=dtype)
        self.conv4 = ConvBN(64, 128, 3, 2, bias=bias, device=device, dtype=dtype)
        self.res2 = ndl.nn.Residual(
            ndl.nn.Sequential(
                ConvBN(128, 128, 3, 1, bias=bias, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, bias=bias, device=device, dtype=dtype)
            )
        )
        self.flatten = ndl.nn.Flatten()
        self.linear = ndl.nn.Linear(128, 128, bias=bias, device=device, dtype=dtype)
        self.relu = ndl.nn.ReLU()
        self.linear2 = ndl.nn.Linear(128, 10, bias=bias, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x) # (seq_len, bs, embedding_size)
        out, h = self.model(x, h)
        seq_len, bs, hidden_size = out.shape
        out = out.reshape((seq_len * bs, hidden_size))
        out = self.linear(out)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)