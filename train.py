import tqdm
import torch
from torch.autograd import Variable

MODEL_PATH = '/Users/lixiaoyang/Desktop/CS4486/HW_3 (1)/'

def train(epoch_n, batch_size, device, model, criterion, optimizer, train_loader, test_loader):
    print('training...')
    for epoch in range(0, epoch_n):
        # train
        train_loss = 0
        train_acc = 0
        train_step = 0
        model.train()
        for image, label in tqdm.tqdm(train_loader):
            image = Variable(image.to(device))
            label = Variable(label.to(device))
            output = model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / batch_size
            train_acc += acc
            train_step += 1
        train_loss /= train_step
        train_acc /= train_step
        # eval
        eval_loss = 0
        eval_acc = 0
        eval_step = 0
        model.eval()
        for image, label in test_loader:
            image = Variable(image.to(device))
            label = Variable(label.to(device))
            output = model(image)
            loss = criterion(output, label)
            eval_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / batch_size
            eval_acc += acc
            eval_step += 1
        eval_loss /= eval_step
        eval_acc /= eval_step
        # print the loss and accuracy
        print('[{:3d}/{:3d}] Train Loss: {:11.9f} | Train Accuracy: {:6.4f} | Eval Loss: {:11.9f} | Eval Accuracy: {:6.4f}'
            .format(epoch, epoch_n, train_loss, train_acc, eval_loss, eval_acc))
        # save the model
        torch.save(model.state_dict(), MODEL_PATH + 'epoch{}.pth'.format(epoch))

    print('finish training...')