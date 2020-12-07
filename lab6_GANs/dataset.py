from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision


def get_pokemon_dataset() -> DataLoader:
    d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                               'c065c0e2593b8b161a2d7873e42418bf6a21106c')
    data_dir = d2l.download_extract('pokemon')
    pokemon = torchvision.datasets.ImageFolder(data_dir)

    batch_size = 256
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    pokemon.transform = transformer

    data_loader = torch.utils.data.DataLoader(
        pokemon, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader


def plot_original_images():
    data_loader = get_pokemon_dataset()
    d2l.set_figsize((4, 4))
    num_rows = 4
    num_cols = 5
    scale = 1.5
    for X, y in data_loader:
        imgs = X[0:20, :, :, :].permute(0, 2, 3, 1) / 2 + 0.5
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            ax.imshow(d2l.numpy(img))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        plt.savefig("original.png")
        break


if __name__ == '__main__':
    # get_pokemon_dataset()
    plot_original_images()
