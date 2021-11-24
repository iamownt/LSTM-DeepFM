import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    """Create a Two Layer LSTMEncoder for the MaskModel。
    Args:
        input_dim: the input feature_size
        hidden_dim: the hidden size
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, step):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.step = step
        self.cell1 = nn.LSTMCell(input_dim, hidden_dim, bias=True)
        self.cell2 = nn.LSTMCell(hidden_dim, embedding_dim, bias=True)

    def init_hidden_state(self, batch_size, hidden_dim):
        h = torch.zeros(batch_size, hidden_dim).to(device)
        c = torch.zeros(batch_size, hidden_dim).to(device)
        return h, c

    def forward(self, time_series):
        """
        :param time_series: the expected time_series shape is (batch_size, time_step, feature_size)
        :return: h: last cell's hidden state
        """
        batch_size = time_series.size(0)
        h0, c0 = self.init_hidden_state(batch_size, self.hidden_dim)
        h1, c1 = self.init_hidden_state(batch_size, self.embedding_dim)

        for i in range(self.step):
            h0, c0 = self.cell1(time_series[:, i, :], (h0, c0))
            h1, c1 = self.cell2(h0, (h1, c1))

        return h1


class LSTMDecoder(nn.Module):
    """Create a Two Layer LSTMDecoder"""

    def __init__(self, input_dim, hidden_dim, embedding_dim, step):
        super(LSTMDecoder, self).__init__()
        assert(embedding_dim < input_dim * step)  # or the task will be so easy
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.step = step
        self.cell1 = nn.LSTMCell(embedding_dim, hidden_dim, bias=True)
        self.cell2 = nn.LSTMCell(hidden_dim, input_dim, bias=True)

    def init_hidden_state(self, batch_size, hidden_dim):
        h = torch.zeros(batch_size, hidden_dim).to(device)
        c = torch.zeros(batch_size, hidden_dim).to(device)
        return h, c

    def forward(self, emb_inp):
        # Assume emb_inp has shape (batch_size, embedding_size), then expand it.
        recon_lis = []
        emb_inp = emb_inp.unsqueeze(1).repeat(1, self.step, 1)
        batch_size = emb_inp.size(0)
        h0, c0 = self.init_hidden_state(batch_size, self.hidden_dim)
        h1, c1 = self.init_hidden_state(batch_size, self.input_dim)

        for i in range(self.step):
            h0, c0 = self.cell1(emb_inp[:, i, :], (h0, c0))
            h1, c1 = self.cell2(h0, (h1, c1))
            recon_lis.append(h1)

        return torch.stack(recon_lis).permute(1, 0, 2)


class LSTMEncoderDecoder(nn.Module):
    """
    Use simple two layer LSTM.
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, step):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, embedding_dim, step)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, embedding_dim, step)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def check_lstm():
    batch_size, time_step, feature_size, hidden_size, embedding_size = 3, 4, 13, 24, 12
    dummpy_data = torch.rand(batch_size, time_step, feature_size).to(device)
    lstm_ende = LSTMEncoderDecoder(feature_size, hidden_size, embedding_size, time_step).to(device)
    print("The input: ", dummpy_data)
    print("The reconstruct: ", lstm_ende(dummpy_data))


if __name__ == "__main__":
    check_lstm()