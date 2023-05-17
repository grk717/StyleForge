from segmentation_utils import convert_dataset

import click

@click.group()
def cli():
    click.echo("Starting cli")

@cli.command()
@click.option('--only_val', is_flag=True, help='perform preprocessing only for validation dataset')
@click.option('--val_ann_path', default="C:\\Users\\grk\\git\\StyleForge\\data\\validation\\annos", help='val annotation folder path')
@click.option('--train_ann_path', default="C:\\Users\\grk\\git\\StyleForge\\data\\train\\annos", help='train annotation folder path')
@click.option('--val_out_path', default="C:\\Users\\grk\\git\\StyleForge\\data\\validation\\df.csv", help='where to store the csv for val')
@click.option('--train_out_path', default="C:\\Users\\grk\\git\\StyleForge\\data\\train\\df.csv", help='where to store the csv for train')
def preprocess_data(**kwargs):
    convert_dataset(kwargs['val_ann_path'], kwargs['val_out_path'])
    if not kwargs['only_val']:
        convert_dataset(kwargs['train_ann_path'], kwargs['train_out_path'])


if __name__ == "__main__":
    cli() 
