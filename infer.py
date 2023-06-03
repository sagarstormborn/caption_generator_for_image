
import torch.optim as optim
from utils import load_checkpoint, print_examples
from data_loader import *
from model import CNNtoRNN


def infer():

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    train_CNN = False

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        "Data/Images/",
        "Data/captions.txt",
        transform=transform,
        num_workers=6,
    )

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4


    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    print_examples(model,device,dataset)

if __name__ == "__main__":
    infer()
