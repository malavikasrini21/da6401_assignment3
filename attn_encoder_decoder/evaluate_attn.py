import torch
from torch.utils.data import DataLoader
from dataload import get_dataloaders
from model_attn import Encoder, Decoder, Seq2Seq
import argparse
import wandb  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

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
    return ''.join(chars)


def load_model(model_path, device, src_vocab_size, tgt_vocab_size, config, load_partial=False):
    encoder = Encoder(src_vocab_size, config['embed_size'], config['hidden_size'],
                      config['num_layers'], config['rnn_cell'], config['dropout'])
    decoder = Decoder(tgt_vocab_size, config['embed_size'], config['hidden_size'],
                      config['num_layers'], config['rnn_cell'], config['dropout'])

    model = Seq2Seq(encoder, decoder, device, config['tgt_vocab']).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if load_partial:
        model_state = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        print(f"Loading {len(filtered_state_dict)} / {len(checkpoint)} parameters from checkpoint.")
        model_state.update(filtered_state_dict)
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def plot_grid_heatmaps(attention_list, input_list, output_list):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import wandb
    from matplotlib import font_manager

    # Load Devanagari font
    devanagari_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf")

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(attention_list):
            break
        
        attn = attention_list[i]
        num_output = len(output_list[i])
        num_input = len(input_list[i])

        # Clip attention matrix to match label lengths
        attn = attn[:num_output, :num_input]

        sns.heatmap(attn.cpu().detach().numpy(),
            ax=ax,
            xticklabels=input_list[i],
            yticklabels=output_list[i],
            cmap="viridis")


        ax.set_xticklabels(input_list[i], fontproperties=devanagari_font, rotation=90)
        ax.set_yticklabels(output_list[i], fontproperties=devanagari_font, rotation=0)

        ax.set_title(f"Sample {i}", fontproperties=devanagari_font, fontsize=12)
        ax.set_xlabel("Input", fontproperties=devanagari_font, fontsize=10)
        ax.set_ylabel("Output", fontproperties=devanagari_font, fontsize=10)

    plt.tight_layout()
    wandb.log({"attention_grid": wandb.Image(fig, caption="Attention Heatmaps")})
    plt.close(fig)

def evaluate_all(model, dataloader, device, beam_width, max_len, src_vocab, tgt_vocab):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi['<pad>'])
    total_loss = 0
    total_correct = 0
    total_chars = 0
    word_correct = 0
    word_total = 0
    results = []

    failure_attns = []
    failure_srcs = []
    failure_preds = []

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            outputs, attentions = model(src, tgt, src_lens, teacher_forcing_ratio=0, return_attentions=True)

            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, tgt_flat)
            total_loss += loss.item() * src.size(0)

            for i in range(src.size(0)):
                src_seq = src[i]
                tgt_seq = tgt[i]
                src_len = src_lens[i]
                tgt_len = tgt_lens[i]

                pred_str = model.beam_search_decode(src_seq, src_len, beam_width=beam_width, max_len=max_len)
                tgt_str = tokens_to_string(tgt_seq[1:tgt_len-1], tgt_vocab)
                src_str = tokens_to_string(src_seq[:src_len], src_vocab)

                results.append((src_str, tgt_str, pred_str))

                # Char and word accuracy
                min_len = min(len(tgt_str), len(pred_str))
                total_correct += sum(1 for a, b in zip(tgt_str[:min_len], pred_str[:min_len]) if a == b)
                total_chars += tgt_len - 2
                if pred_str == tgt_str:
                    word_correct += 1
                else:
                    if len(failure_attns) < 9:
                        failure_attns.append(attentions[i].squeeze(0))  # [tgt_len, src_len]
                        failure_srcs.append(list(src_str))
                        failure_preds.append(list(pred_str))
                word_total += 1

    if len(failure_attns) > 0:
        plot_grid_heatmaps(failure_attns, failure_srcs, failure_preds)

    avg_loss = total_loss / len(dataloader.dataset)
    char_accuracy = total_correct / total_chars if total_chars > 0 else 0.0
    word_accuracy = word_correct / word_total if word_total > 0 else 0.0

    return results, char_accuracy, word_accuracy


def main(args):
    wandb.init(project="da6401_assignment3", name="test-run", config=vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, src_vocab, tgt_vocab = get_dataloaders(args.dataset_path, batch_size=1)

    config = {
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'rnn_cell': args.rnn_cell,
        'tgt_vocab': tgt_vocab
    }

    model = load_model(args.model_path, device, len(src_vocab), len(tgt_vocab), config, load_partial=True)
    model.to(device)
    results, char_acc, word_acc = evaluate_all(model, test_loader, device, args.beam_width, args.max_len, src_vocab, tgt_vocab)

    df = pd.DataFrame([
        {"src": src, "target": tgt, "pred": pred, "status": "pred_true" if pred == tgt else "pred_false"}
        for src, tgt, pred in results
    ])
    df.to_csv("decoded_output.tsv", sep="\t", index=False)

    print({"char_level_accuracy": char_acc})
    print({"word_level_accuracy": word_acc})
    wandb.log({
        "char_level_accuracy": char_acc,
        "word_level_accuracy": word_acc,
        "predictions_table": wandb.Table(dataframe=df)
    })

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
