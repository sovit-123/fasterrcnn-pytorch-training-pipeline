from torch_utils.engine import (
    train_one_epoch, evaluate
)
from config import (
    DEVICE, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS,
    OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.fasterrcnn_mobilenetv3_large_fpn import create_model
from custom_utils import (
    save_model, 
    save_train_loss_plot,
    Averager, show_tranformed_image
)

import torch

if __name__ == '__main__':
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader)

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # LR will be zero as we approach `steps` number of epochs each time.
    # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
    steps = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=steps,
        T_mult=1,
        verbose=True
    )

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler
        )

        evaluate(model, valid_loader, device=DEVICE)

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)

        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_train_loss_plot(OUT_DIR, train_loss_list)