import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [B, H] -> the current decoder hidden state
        # encoder_outputs: [B, T, H]
        T = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, T, 1)  # [B, T, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        energy = energy @ self.v  # [B, T]
        
        energy.masked_fill_(mask == 0, -1e10)
        attention = F.softmax(energy, dim=1)  # [B, T]
        return attention

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers, rnn_cell, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_size)
        self.rnn_type = rnn_cell.lower()
        self.rnn = get_rnn(rnn_cell)(embed_size, hidden_size, num_layers,
                                     dropout=dropout, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths,
                                                   batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden  # outputs for attention

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, hidden_size, num_layers, rnn_cell, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_size)
        self.rnn = get_rnn(rnn_cell)(embed_size + hidden_size, hidden_size, num_layers,
                                     dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.rnn_cell_type = rnn_cell.lower()

    def forward(self, input_char, hidden, encoder_outputs, mask):
        # input_char: [B], hidden: [num_layers, B, H], encoder_outputs: [B, T, H]
        embedded = self.embedding(input_char).unsqueeze(1)  # [B, 1, E]

        if isinstance(hidden, tuple):  # LSTM
            dec_hidden = hidden[0][-1]
        else:
            dec_hidden = hidden[-1]

        attn_weights = self.attention(dec_hidden, encoder_outputs, mask)  # [B, T]
        context = attn_weights.unsqueeze(1) @ encoder_outputs  # [B, 1, H]

        rnn_input = torch.cat((embedded, context), dim=2)  # [B, 1, E+H]
        output, hidden = self.rnn(rnn_input, hidden)       # [B, 1, H], hidden: same shape

        output = output.squeeze(1)
        context = context.squeeze(1)
        prediction = self.fc(torch.cat((output, context), dim=1))  # [B, output_dim]
        return prediction, hidden, attn_weights

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

    def create_mask(self, src):
        return (src != self.pad_idx).to(self.device)

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5, return_attentions=True):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc.out_features).to(self.device)
        all_attentions = [] if return_attentions else None

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
        input_char = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            output, hidden, attn_weights = self.decoder(input_char, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            if return_attentions:
                all_attentions.append(attn_weights.detach().cpu())
            top1 = output.argmax(1)
            input_char = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        if return_attentions:
            # all_attentions: list of [B, T] -> convert to [B, tgt_len-1, src_len]
            attentions = torch.stack(all_attentions, dim=1)
            return outputs, attentions
        return outputs


    def beam_search_decode(self, src, src_len, beam_width=3, max_len=30):
        encoder_outputs, hidden = self.encoder(src.unsqueeze(0), [src_len])
        mask = self.create_mask(src.unsqueeze(0))
        input_char = torch.tensor([self.sos_idx], device=self.device)

        sequences = [[[], 0.0, hidden]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, h in sequences:
                decoder_output, new_hidden, _ = self.decoder(input_char, h, encoder_outputs, mask)
                log_probs = torch.log_softmax(decoder_output, dim=1)
                topk_probs, topk_idx = log_probs.topk(beam_width)

                for i in range(beam_width):
                    next_token = topk_idx[0][i].item()
                    next_score = score + topk_probs[0][i].item()
                    all_candidates.append([seq + [next_token], next_score, new_hidden])

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
            input_char = torch.tensor([sequences[0][0][-1]], device=self.device)

            if sequences[0][0][-1] == self.eos_idx:
                break

        best_sequence = sequences[0][0]
        return ''.join([
            self.idx2char[idx] for idx in best_sequence
            if idx not in (self.pad_idx, self.sos_idx, self.eos_idx)
        ])
