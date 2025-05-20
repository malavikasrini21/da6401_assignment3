import torch
import os
import wandb
from tqdm import tqdm
from model_attn import Encoder, Decoder, Seq2Seq
from dataload import get_dataloaders
from viss2 import create_interactive_connectivity


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


def load_model(model_path, device, config, src_vocab_size, tgt_vocab_size):
    encoder = Encoder(src_vocab_size, config['embed_size'], config['hidden_size'],
                      config['num_layers'], config['rnn_cell'], config['dropout'])
    decoder = Decoder(tgt_vocab_size, config['embed_size'], config['hidden_size'],
                      config['num_layers'], config['rnn_cell'], config['dropout'])
    model = Seq2Seq(encoder, decoder, device, config['tgt_vocab']).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate_and_visualize(model, dataloader, device, input_vocab, output_vocab, max_examples=3):
    table = wandb.Table(columns=["Input (Latin)", "Prediction (Devanagari)", "Attention (HTML)"])
    model.eval()
    examples_logged = 0

    with torch.no_grad():
        for batch_idx, (src, tgt, src_lens, tgt_lens) in enumerate(tqdm(dataloader, desc="Visualizing")):
            src, tgt = src.to(device), tgt.to(device)

            # Get output + attention
            logits, attentions = model(src, tgt, src_lens, teacher_forcing_ratio=0.0, return_attentions=True)
            pred_tokens = logits.argmax(-1)

            for i in range(src.size(0)):
                if examples_logged >= max_examples:
                    break

                src_str = tokens_to_string(src[i][:src_lens[i]], input_vocab)
                pred_str = tokens_to_string(pred_tokens[i], output_vocab)
                src_tokens = list(src_str)
                tgt_tokens = list(pred_str)
                attn = attentions[i].cpu()

                html_file = create_interactive_connectivity(
                    attn_matrix=attn,
                    input_seq=src_tokens,
                    output_seq=tgt_tokens,
                    filename=f"attention_{batch_idx}_{i}.html"
                )

                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                wandb.log({f"attention_html_{batch_idx}_{i}": wandb.Html(html_content)})
                table.add_data(src_str, pred_str, wandb.Html(html_content))

                os.remove(html_file)
                examples_logged += 1

        wandb.log({"Interactive_Attention_Examples": table})


def main(args):
    wandb.init(project="attention-visualization", name="html-hover-visualization", config=vars(args))
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

    model = load_model(args.model_path, device, config, len(src_vocab), len(tgt_vocab))
    evaluate_and_visualize(model, test_loader, device, src_vocab, tgt_vocab, max_examples=args.max_examples)

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument('--max_examples', type=int, default=4502, help="Max samples to visualize")
    args = parser.parse_args()

    main(args)
