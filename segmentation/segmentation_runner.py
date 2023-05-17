import click
from segmentation_train import run

@click.group()
def cli():
    click.echo("Starting cli")

@cli.command()
@click.option('--epochs', default=50, help='number of epochs')
@click.option('--batch_size', default=8, help='size of a batch for training')
@click.option('--base_lr', default=0.0001, help='starting learing rate')
@click.option('--dataset_type', type=click.Choice(['big', 'mini'], case_sensitive=False), help='which dataset to use')
@click.option('--train_csv', default=r"C:\Users\grk\git\StyleForge\notebooks\deepfashion_val_segm.csv", help="path to train csv")
@click.option('--val_csv', default=r"C:\Users\grk\git\StyleForge\notebooks\deepfashion_val_segm.csv", help="path to val csv")
def segmentation_train(**kwargs):
    run(kwargs)


if __name__ == "__main__":
    cli() 
