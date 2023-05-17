from clearml import Task
from config import TrainConfig
import pandas as pd
from segmentation_utils import *
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch

from torch.utils.data import DataLoader

def run(opt):
    epochs = opt["epochs"]
    batch_size = opt["batch_size"]
    base_lr = opt["base_lr"]

    config = TrainConfig()
    
    task_name = f"{config.MODEL}_{config.ENCODER}_{config.ACTIVATION}_{base_lr}"
    task = Task.init(project_name="styleforge/segmentation", task_name=task_name)
    #model_storage = f"/mnt/tank/scratch/pgrinkevich/models/{task.task_id}"
    model_storage = f"models/"
    if not os.path.exists(model_storage):
        os.mkdir(model_storage)  # create model storage
    logger = task.get_logger()

    
    model = smp.create_model(config.MODEL,
        encoder_name=config.ENCODER, 
        encoder_weights=config.ENCODER_WEIGHTS, 
        classes=len(config.CLASSES), 
        activation=config.ACTIVATION,
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    loss = smp.losses.DiceLoss(mode="multilabel", from_logits=False if config.ACTIVATION else True)
    loss.__name__ = 'dice_loss' # smp bug workaround
    print(loss.from_logits)
    metrics = [
        smp.utils.metrics.IoU(),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=base_lr),
    ])
    
    if opt["dataset_type"] == "mini":
        df = pd.read_csv(opt["val_csv"])
        df_train, df_val = train_test_split(df, train_size=config.TRAIN_CLASSES_FRAC, random_state=config.SEED, stratify=df["category_id"])
    else:
        df_train = pd.read_csv(opt["train_csv"])
        df_val = pd.read_csv(opt["val_csv"])

    train_dataset = DeepFashionDataset(
        df_train,
        classes=config.CLASSES,
        transform=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = DeepFashionDataset(
        df_val, 
        classes=config.CLASSES,
        transform=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    vis_dataset = DeepFashionDataset(
        df_val, 
        classes=config.CLASSES,
        transform=get_validation_augmentation()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=config.DEVICE,
        verbose=True,
    )

    max_score = 0

    for epoch in range(1, epochs):
        
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        logger.report_scalar(title="dice_loss", series='', iteration=epoch, value=train_logs["dice_loss"])
        logger.report_scalar(title="iou_score", series='', iteration=epoch, value=train_logs["iou_score"])
        
        with torch.no_grad():
            for i in range(3):
                n = np.random.choice(len(valid_dataset))
                image_vis = cv2.cvtColor(vis_dataset[n][0].astype('uint8'), cv2.COLOR_BGR2RGB)
                image, gt_mask = valid_dataset[n]
                gt_mask = gt_mask.squeeze()
                x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
                model.to(config.DEVICE)
                pr_mask = model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy())
                pr_mask[pr_mask < 0.5] == 0
                logger.report_image("images", "source", iteration=epoch, image=image_vis)
                logger.report_image("images", "gt_mask", iteration=epoch, image=np.argmax(gt_mask, axis=0) * ((255 / (len(config.CLASSES) - 1)) - 1))
                logger.report_image("images", "predict", iteration=epoch, image=np.argmax(pr_mask, axis=0) * ((255 / (len(config.CLASSES) - 1)) - 1))
                
        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        
        if epoch % 10 == 0:
            torch.save(model, f"{model_storage}/model_ckpt_epoch_{epoch}.pth")
