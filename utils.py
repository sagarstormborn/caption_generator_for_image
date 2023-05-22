import torch
import torchvision.transforms as transforms
from PIL import Image
from chat_gpt_api import *

def print_examples(model, device, dataset):
    api_key = read_api_key('api_key.txt')
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img3 = transform(Image.open("examples/car.jpg").convert("RGB")).unsqueeze(
        0
    )
    text = " ".join(model.caption_image(test_img3.to(device), dataset.vocab)[1:-1])
    print("car.jpg GT: A car splashes through mud and leaves on the forest floor")
    print(
        "car.jpg OUTPUT: "
        + text
    )
    askGPT(text, api_key)
    test_img2 = transform(
        Image.open("examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    text =" ".join(model.caption_image(test_img2.to(device), dataset.vocab)[1:-1])
    print("child.jpg GT: Child holding red frisbee outdoors")
    print(
        "child.jpg OUTPUT: "
        + text
    )
    askGPT(text, api_key)

    test_img1 = transform(Image.open("examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    text = " ".join(model.caption_image(test_img1.to(device), dataset.vocab)[1:-1])
    print("dog.jpg GT: Dog on a beach by the ocean")
    print(
        "dog.jpg OUTPUT: "
        + text
    )

    askGPT(text, api_key)
    test_img4 = transform(
        Image.open("examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    text=" ".join(model.caption_image(test_img4.to(device), dataset.vocab)[1:-1])
    print("boat.png GT: A small boat in the ocean")
    print(
        "boat.png OUTPUT: "
        + text
    )
    askGPT(text, api_key)
    test_img5 = transform(
        Image.open("examples/football.jpg").convert("RGB")
    ).unsqueeze(0)
    text=" ".join(model.caption_image(test_img5.to(device), dataset.vocab)[1:-1])
    print("football.jpg GT: A red team and a white team are playing football ")
    print(
        "football.jpg OUTPUT: "
        + text
    )
    askGPT(text, api_key)
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step