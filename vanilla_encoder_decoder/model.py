import torch
import torch.nn as nn

def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == "lstm":
        return nn.LSTM
    elif rnn_type == "gru":
        return nn.GRU
    elif rnn_type == "rnn":
        return nn.RNN
    else:
        raise ValueError("Unsupported RNN type: choose from 'RNN', 'LSTM', 'GRU'")

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers, rnn_cell, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_size)
        self.rnn_type = rnn_cell.lower()
        self.rnn = get_rnn(rnn_cell)(embed_size, hidden_size, num_layers,
                                     dropout=dropout, batch_first=True)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths,
                                                   batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, hidden_size, num_layers, rnn_cell, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_size)
        self.rnn = get_rnn(rnn_cell)(embed_size, hidden_size, num_layers,
                                     dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.output_dim = output_dim

    def forward(self, input_char, hidden):
        embedded = self.embedding(input_char).unsqueeze(1)  # (B, 1, E)
        output, hidden = self.rnn(embedded, hidden)         # output: (B, 1, H)
        prediction = self.fc(output.squeeze(1))             # (B, output_dim)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, tgt_vocab):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.tgt_vocab = tgt_vocab
        self.idx2char = tgt_vocab.itos
        self.pad_idx = tgt_vocab.stoi["<pad>"]
        self.sos_idx = tgt_vocab.stoi["<sos>"]
        self.eos_idx = tgt_vocab.stoi["<eos>"]

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.output_dim).to(self.device)

        hidden = self.encoder(src, src_lengths)
        input_char = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_char, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_char = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

    def beam_search_decode(self, src, src_len, beam_width=3, max_len=30):
        hidden = self.encoder(src.unsqueeze(0), [src_len])
        input_char = torch.tensor([self.sos_idx], device=self.device)

        sequences = [[[], 0.0, hidden]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, h in sequences:
                decoder_output, new_hidden = self.decoder(input_char, h)
                log_probs = torch.log_softmax(decoder_output, dim=1)
                topk_probs, topk_idx = log_probs.topk(beam_width)

                for i in range(beam_width):
                    next_token = topk_idx[0][i].item()
                    next_score = score + topk_probs[0][i].item()
                    all_candidates.append([seq + [next_token], next_score, new_hidden])

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
            input_char = torch.tensor([sequences[0][0][-1]], device=self.device)

            # Stop if <eos> is reached
            if sequences[0][0][-1] == self.eos_idx:
                break

        best_sequence = sequences[0][0]
        return ''.join([
            self.idx2char[idx] for idx in best_sequence
            if idx not in (self.pad_idx, self.sos_idx, self.eos_idx)
        ])
