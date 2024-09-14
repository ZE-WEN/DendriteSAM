import os
import numpy as np

import torch

import torch_em
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

def get_dataloader(split, patch_shape, batch_size, train_instance_segmentation):
    image_dir = 'img_dir2'
    segmentation_dir = 'mask_dir2'
    raw_key, label_key = "*.tif", "*.tif" # input data type, all tif files

    if split == "train":
        roi = np.s_[:99, :, :]
    else:
        roi = np.s_[99:, :, :] 

    if train_instance_segmentation:

        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        label_transform=label_transform,
        num_workers=2, shuffle=True, raw_transform=sam_training.identity,
    )
    return loader


def run_training(checkpoint_name, model_type, train_instance_segmentation):
    """Run the actual model training."""

    # hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 1024, 1024)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled,by default 25
                              # if the number of objects is smaller, it will take the actual number of objects
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Available device:',device)

    # load data
    train_loader = get_dataloader("train", patch_shape, batch_size, train_instance_segmentation)
    val_loader = get_dataloader("val", patch_shape, batch_size, train_instance_segmentation)

    # Run training.
    sam_training.training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        early_stopping=100,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path='work_dir/MicroSAM/vit_l.pt', #if you want to fine-tune from other models
                                                                     #instead of the original SAM models, specify the path here
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
        save_every_kth_epoch=5,
    )


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used later
    export_path = "./finetuned_dendrite_model_em_downsample_vit_l.pth" # final model name
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """Finetune a Segment Anything model.
    """
    # backnone model type
    model_type = "vit_l"
    # The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_dendrite_vit_l_downsample"
    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True
    run_training(checkpoint_name, model_type, train_instance_segmentation)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()