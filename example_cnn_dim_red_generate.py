import torch
import torchvision
import torchvision.transforms as transforms
from example_cnn_dim_red_model import Encoder, Decoder
import matplotlib.pyplot as plt
import numpy as np

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 256

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True)

### Dimension of the latent space
latent_dimension = 4


### Model definition
encoder = Encoder(encoded_space_dim=latent_dimension)
decoder = Decoder(encoded_space_dim=latent_dimension)


### Loading the learnt models
PATH = './dim_red_encoder.pth'
encoder.load_state_dict(torch.load(PATH))

PATH = './dim_red_decoder.pth'
decoder.load_state_dict(torch.load(PATH))

encoder.eval()
decoder.eval()

with torch.no_grad():
    # calculate mean and std of latent code, generated takining in test images as inputs
    images, labels = iter(test_loader).next()
    images = images.to(device)
    latent = encoder(images)
    latent = latent.cpu()

    mean = latent.mean(dim=0)
    print(mean)
    std = (latent - mean).pow(2).mean(dim=0).sqrt()
    print(std)

    # sample latent vectors from the normal distribution
    latent = torch.randn(128, latent_dimension)*std + mean

    # reconstruct images from the random latent vectors
    latent = latent.to(device)
    img_recon = decoder(latent)
    img_recon = img_recon.cpu()

    fig, ax = plt.subplots(figsize=(20, 8.5))
    show_image(torchvision.utils.make_grid(img_recon[:100],10,5))
    plt.show()