import torch
from torch import mps
import torch.multiprocessing as mp
from discriminator import Discriminator
from generator import Generator
from initialize import initialize_weights
from art_dataset import AbstractArtDataset
from torch import nn
import wandb
from torch.utils.data import DataLoader


def train_step():
    gen_loss = 0
    dis_loss = 0
    for step, data in enumerate(train_loader):
        # generate the labels
        real_samples = data.to(device=device)
        real_labels = torch.ones((real_samples.size(0), 1)).to(device=device)
        generated_labels = torch.zeros(
            (real_samples.size(0), 1)).to(device=device)

        # generate latent space noise
        # generate the constant matrix as required by Style GAN
        const = torch.zeros(
            size=(real_samples.size(0), 512, 4, 4)).to(device=device)
        latent_space_samples = torch.randn(
            (real_samples.size(0), 512)).to(device=device)
        generated_samples = generator(latent_space_samples, const)

        # Train the discriminator
        discriminator.zero_grad()
        out_gen_disc = discriminator(
            generated_samples).view(real_samples.size(0), 1)
        out_real_disc = discriminator(
            real_samples).view(real_samples.size(0), 1)

        gen_loss = loss_function(out_gen_disc, generated_labels)
        dis_loss = loss_function(out_real_disc, real_labels)

        discriminator_loss = (gen_loss+dis_loss)/2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        latent_space_samples = torch.randn(
            (real_samples.size(0), 100)).to(device=device)
        generated_samples = generator(latent_space_samples)

        # Training the generator
        generator.zero_grad()
        output_discriminator_generated = discriminator(
            generated_samples).view(real_samples.size(0), 1)
        generator_loss = loss_function(
            output_discriminator_generated, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        gen_loss += generator_loss.item()
        dis_loss += discriminator_loss.item()

        del output_discriminator_generated
        del real_labels
        del real_samples
        del generated_labels
        del generated_samples
        del latent_space_samples
        del out_gen_disc
        del out_real_disc

        mps.empty_cache()

    gloss = gen_loss/train_steps
    dloss = dis_loss/train_steps

    return gloss, dloss


def training_loop():
    discriminator.train(True)
    for epoch in range(num_epochs):

        generator.train(True)  # train mode

        train_losses = train_step()

        generator.eval()  # eval mode

        print('Epoch {epoch}'.format(epoch=epoch+1))
        print("Generator Loss: {gloss}".format(gloss=train_losses[0]))
        print("Discriminator Loss: {dloss}".format(dloss=train_losses[1]))

        wandb.log(
            {'Generator Loss': train_losses[0], 'Discriminator Loss': train_losses[1]})

        # save model at epoch checkpoints
        if ((epoch+1) % 5 == 0):
            pathgen = 'weights/generator/generator{number}.pth'.format(
                number=epoch+1)
            pathdis = 'weights/discriminator/discriminator{number}.pth'.format(
                number=epoch+1)
            torch.save(generator.state_dict(), pathgen)
            torch.save(discriminator.state_dict(), pathdis)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    # initial setup
    ids = list(range(0, 2782))

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    dataset = AbstractArtDataset(ids)

    wandb.init(
        project="Style-GAN-Implementation",
        config={
            "learning_rate": 0.0002,
            "architecture": "Style GAN",
            "dataset": "Abstract art",
        },
    )

    train_loader = DataLoader(dataset, **params)

    # device usage
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # get the models
    generator = Generator().to(device=device)
    discriminator = Discriminator().to(device=device)

    initialize_weights(generator)
    initialize_weights(discriminator)

    # hyperparameters
    lr = 0.0002
    num_epochs = 1000
    loss_function = nn.BCEWithLogitsLoss()

    # set optimizer
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    train_steps = (len(ids)+params['batch_size']-1)//params['batch_size']

    training_loop()
