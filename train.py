import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from data_loader import get_loader
from model import CNNtoRNN

def train():
    transform=transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]
    )

    train_loader, dataset= get_loader(
        "Data/Images/",
        "Data/captions.txt",
        transform=transform,
        num_workers=6,
        batch_size=32)

    torch.backends.cudnn.benchmark=True # cuDNN optimization
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model=False
    save_model=False
    train_CNN=False

    embed_size=256
    hidden_size=256
    vocab_size=len(dataset.vocab)
    learning_rate=3e-4
    num_epochs=100

    writer=SummaryWriter("runs/flickr")
    step =0

    model=CNNtoRNN(embed_size,hidden_size,vocab_size,num_epochs).to(device)
    criterion =nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer= optim.Adam(model.parameters(),lr=learning_rate)

    for name ,param in model.encoderCNN.incepton.named_parameters():
        if "fc.weight" in name or "fc.bais" in name:
            param.requires_grad=True
        else :
            param.requires_grad=train_CNN

    if load_model:
        step=load_checkpoint(torch.load("my_checkpoint.pth.tar"),model,optimizer)

    model.train()

    for epoch in range(num_epochs):
        if save_model:
            checkpoint={
                "state_dict":model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":step,
            }
            save_checkpoint(checkpoint)
            for idx,(imgs,captions) in tqdm(
                enumerate(train_loader),total=len(train_loader),leave=False
            ):
                imgs=imgs.to(device)
                captions=captions.to(device)

                outputs=model(imgs,captions[:-1])
                loss=criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))

                writer.add_scalar("Training loss", loss.item(), global_step=step)
                step+=1

                optimizer.zero_grad()
                loss.backward(loss)
                optimizer.step()
if __name__=="__main__":
    train()
