import torch
from torch.utils.data import DataLoader
from dataload import get_dataloaders
from model import Encoder, Decoder, Seq2Seq
import argparse
import wandb  

def tokens_to_string(tokens, vocab):
    chars = []
    for token_id in tokens:
        try:
            char = vocab.itos[token_id.item()]
        except AttributeError:
            char = str(token_id.item())
        if char in ['<pad>', '<sos>', '<eos>']:
            continue
        chars.append(char)
    return ''.join(chars)  # <-- no space between characters


def load_model(model_path, device, src_vocab_size, tgt_vocab_size, config):
    encoder = Encoder(src_vocab_size, config['embed_size'], config['hidden_size'],
                      config['num_layers'], config['rnn_cell'], config['dropout'])
    decoder = Decoder(tgt_vocab_size, config['embed_size'], config['hidden_size'],
                      config['num_layers'], config['rnn_cell'], config['dropout'])
    model = Seq2Seq(encoder, decoder, device, config['tgt_vocab']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_all(model, dataloader, device, beam_width, max_len, src_vocab, tgt_vocab):
    results = []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi['<pad>'])
    total_loss = 0
    total_correct = 0
    total_chars = 0
    word_correct = 0
    word_total = 0

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            for i in range(src.size(0)):
                src_seq = src[i]
                tgt_seq = tgt[i]
                src_len = src_lens[i]
                tgt_len = tgt_lens[i]

                pred_str = model.beam_search_decode(src_seq, src_len, beam_width=beam_width, max_len=max_len)
                tgt_str = tokens_to_string(tgt_seq[1:tgt_len - 1], tgt_vocab)  # Remove <sos>/<eos>
                src_str = tokens_to_string(src_seq[:src_len], src_vocab)

                results.append((src_str, tgt_str, pred_str))

                # Optional: compute char-level accuracy
                min_len = min(len(tgt_str), len(pred_str))
                total_correct += sum(1 for a, b in zip(tgt_str[:min_len], pred_str[:min_len]) if a == b)
                total_chars += tgt_len - 2  # exclude <sos> and <eos>
                if pred_str == tgt_str:
                    word_correct += 1
                word_total += 1

    accuracy = total_correct / total_chars if total_chars > 0 else 0.0
    word_accuracy = word_correct / word_total if word_total > 0 else 0.0
    return results, accuracy,word_accuracy



def main(args):
    # Initialize wandb run
    wandb.init(project="da6401_assignment3",name="test-run" ,config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and vocabs
    _, _, test_loader, src_vocab, tgt_vocab = get_dataloaders(args.dataset_path, batch_size=1)
    
    # Config needed to instantiate model (must match training)
    config = {
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'rnn_cell': args.rnn_cell,
        'tgt_vocab': tgt_vocab
    }

    model = load_model(args.model_path, device, len(src_vocab), len(tgt_vocab), config)
    beam_width = args.beam_width
    max_len = args.max_len

    # decoded_tuples = evaluate_all(model, test_loader, device, beam_width, max_len, src_vocab, tgt_vocab)

    results, char_accuracy,word_accuracy = evaluate_all(model, test_loader, device, beam_width, max_len, src_vocab, tgt_vocab)

    output_file = "decoded_output.tsv"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("src\ttarget\tpred\tstatus\n")
        for src_str, tgt_str, pred_str in results:
            status = "pred_true" if pred_str == tgt_str else "pred_false"
            f.write(f"{src_str}\t{tgt_str}\t{pred_str}\t{status}\n")
    
    print({"char_level_accuracy": char_accuracy})
    print({"word_level_accuracy": word_accuracy})
    wandb.log({
    "char_level_accuracy": char_accuracy,
    "word_level_accuracy": word_accuracy
    })
    #wandb.save(output_file)

    wandb.finish()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Base path to dataset folder')
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument('--beam_width', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=30)
    args = parser.parse_args()

    main(args)
