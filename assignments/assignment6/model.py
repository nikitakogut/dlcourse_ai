import torch
import torch.nn as nn

class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, word_emb_dim=100, embeddings=None, freeze=True, 
                       lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()  
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, word_emb_dim)   
        self.lstm = nn.LSTM(word_emb_dim, lstm_hidden_dim, num_layers=lstm_layers_count)
        self.fc = nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, _ = self.lstm(emb)
        out = self.fc(output)
        return out


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, word_emb_dim=100, embeddings=None, freeze=True, 
                       lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()  
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, word_emb_dim)   
        self.lstm_forward = nn.LSTM(word_emb_dim, lstm_hidden_dim // 2, num_layers=lstm_layers_count)
        self.lstm_backward = nn.LSTM(word_emb_dim, lstm_hidden_dim // 2, num_layers=lstm_layers_count)
        self.fc = nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        output_forward, _ = self.lstm_forward(emb)
        output_backward, _ = self.lstm_backward(torch.flip(emb, (0,)))
        output = torch.stack((output_forward, torch.flip(output_backward, (0,))), 2)
        #output.shape[2]*output.shape[3] must be lstm_hidden_dim
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2]*output.shape[3])
        out = self.fc(output)
        return out

