from datasets import CourtDataset
from torch.utils.data import DataLoader
from tracknet import CourtTrackNet
from general import train, validate
import torch
import os

from tensorboardX import SummaryWriter

if __name__ == "__main__":

    NUM_EPOCHS = 150
    BATCH_SIZE = 4
    LEARNING_RATE = 1.0

    train_dataset = CourtDataset('train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = CourtDataset('val')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = CourtTrackNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    exps_path = 'exps'
    plots_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    log_writer = SummaryWriter('exps/plots')
    model_last_path = os.path.join(exps_path, 'model_last.pth')
    model_best_path = os.path.join(exps_path, 'model_best.pth')
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    val_best_accuracy = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        print('train_loss = {}'.format(train_loss))
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)

        if (epoch > 0) and (epoch % 20 == 0):
            val_loss, precision, accuracy = validate(model, val_loader, criterion, device, epoch)
            print('val_loss = {}'.format(val_loss))
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/accuracy', accuracy, epoch)
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)