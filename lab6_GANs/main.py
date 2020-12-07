import matplotlib.pyplot as plt
from d2l.torch import d2l

from dataset import *
from gan import *


def train(net_D,  net_G,  data_loader,  num_epochs,  lr,  latent_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)

    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hyperparams = {
        'lr': lr,
        'betas': [0.5, 0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hyperparams)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hyperparams)

    animator = d2l.Animator(
        xlabel='epoch', ylabel='loss',
        xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
        legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)

    for epoch in range(1, num_epochs + 1):
        print("Epoch", epoch)
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_loader:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)

        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)

        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
                for i in range(len(fake_x) // 7)], dim=0)

        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)

        plt.savefig(f"results/{epoch}.png")

        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))


if __name__ == '__main__':
    latent_dim, lr, num_epochs = 100, 0.005, 100
    net_D = Discriminator()
    net_G = Generator()
    data_loader = get_pokemon_dataset()
    train(net_D, net_G, data_loader, num_epochs, lr, latent_dim)

