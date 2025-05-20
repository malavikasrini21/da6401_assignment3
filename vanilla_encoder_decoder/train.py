import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
from model import Encoder, Decoder, Seq2Seq
from dataload import *

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'name' : 'vanilla_encoder_decoder_sweep_1',
    'metric': {'name': 'dev_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [0.001, 0.0001]},
        'batch_size': {'values': [32, 64]},
        'embed_size': {'values': [128, 256]},
        'hidden_size': {'values': [256, 512]},
        'num_layers': {'values': [1, 2, 3]},
        'dropout': {'values': [0.2,0.3, 0.5]},
        'rnn_cell': {'values': ['rnn', 'lstm', 'gru']},
        'beam_width': {'values': [1, 3, 5]},
        'epochs': {'value': 10},
        'max_len': {'value': 30}
    }
}

default_config = {
    'lr': 0.001,
    'batch_size': 32,
    'embed_size': 256,
    'hidden_size': 512,
    'num_layers': 3,
    'dropout': 0.3,
    'rnn_cell': 'lstm',
    'beam_width': 3,
    'epochs': 10,
    'max_len': 30
}

def calculate_accuracy(predictions, targets, pad_idx):
    preds = torch.argmax(predictions, dim=-1)
    mask = targets != pad_idx
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def format_run_name(cfg):
    return f"bs_{cfg.batch_size}_hs{cfg.hidden_size}_ems_{cfg.embed_size}_lr_{cfg.lr}_epochs_{cfg.epochs}_rnntype_{cfg.rnn_cell}_num_layers_{cfg.num_layers}"

def train(config=None, dataset_path=None):
    wandb.init(config=config, project="da6401_assignment3")
    config = wandb.config
    run_name = format_run_name(config)
    wandb.run.name=run_name
    wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    # train_set, dev_set, test_set, src_vocab, tgt_vocab = load_datasets(
    #     dataset_path,
    #     #max_len=config.max_len
    #     batch_size=config.batch_size
    # )

    # train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_set, batch_size=config.batch_size)

    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        dataset_path,
        batch_size=config.batch_size
    )

    encoder = Encoder(len(src_vocab), config.embed_size, config.hidden_size,
                      config.num_layers, config.rnn_cell, config.dropout)
    decoder = Decoder(len(tgt_vocab), config.embed_size, config.hidden_size,
                      config.num_layers, config.rnn_cell, config.dropout)
    model = Seq2Seq(encoder, decoder, device, tgt_vocab).to(device)

    # criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi["<pad>"])

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0

        for src, tgt,src_lens,tgt_lens in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt,src_lens)
            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            # total_train_acc += calculate_accuracy(output_flat, tgt_flat, pad_idx=tgt_vocab["<pad>"])
            total_train_acc += calculate_accuracy(output_flat, tgt_flat, pad_idx=tgt_vocab.stoi["<pad>"])


        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)

        model.eval()
        total_dev_loss = 0
        total_dev_acc = 0

        with torch.no_grad():
            for src, tgt,src_lens,tgt_lens in dev_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, src_lens,teacher_forcing_ratio=0.0)
                output_flat = output[:, 1:].reshape(-1, output.shape[-1])
                tgt_flat = tgt[:, 1:].reshape(-1)
                loss = criterion(output_flat, tgt_flat)
                total_dev_loss += loss.item()
                # total_dev_acc += calculate_accuracy(output_flat, tgt_flat, pad_idx=tgt_vocab["<pad>"])
                total_dev_acc += calculate_accuracy(output_flat, tgt_flat, pad_idx=tgt_vocab.stoi["<pad>"])


        avg_dev_loss = total_dev_loss / len(dev_loader)
        avg_dev_acc = total_dev_acc / len(dev_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Dev Loss: {avg_dev_loss:.4f} | Dev Acc: {avg_dev_acc:.4f}")
        wandb.log({
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_acc,
            'dev_loss': avg_dev_loss,
            'dev_accuracy': avg_dev_acc,
            'epoch': epoch + 1
        })

        # Save model if validation accuracy improves
        if avg_dev_acc > best_val_acc:
            best_val_acc = avg_dev_acc
            save_path = f"best_model_{wandb.run.id}.pt"
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run with wandb sweep')
    parser.add_argument('--dataset_path', type=str, default='/data1/malavika/da6401_assgn3/data',
                        help='Base path to the dataset folder')
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="da6401_assignment3")
        wandb.agent(sweep_id, function=lambda: train(dataset_path=args.dataset_path), count=25)
    else:
        train(config=default_config, dataset_path=args.dataset_path)
