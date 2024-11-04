import os
import numpy as np
import torch
import torch_em
from torch_em.transform.label import PerObjectDistanceTransform
import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

def get_dataloader(split, patch_shape, batch_size, train_instance_segmentation):
    image_dir = 'img_dir2'  # change your image directory here
    segmentation_dir = 'mask_dir2'  # change your mask directory here
    raw_key, label_key = "*.tif", "*.tif" # input data type, tif files

    if split == "train":
        roi = np.s_[:80, :, :] # for training
    else:
        roi = np.s_[80:, :, :] # for validation

    if train_instance_segmentation:

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

    
    batch_size = 1  
    patch_shape = (1, 1024, 1024) 
    n_objects_per_batch = 25  
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
        checkpoint_path='vit_b.pt', # initialize the wights, you need to change it when initialzing with your weights
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
        save_every_kth_epoch=5,
    )


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    
    export_path = "./ViT-B-resize-EM-dendrite.pth" # final model name
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """Finetune a Segment Anything model.
    """
    model_type = "vit_b"
    checkpoint_name = "DendriteSAM_vit_b_resize"
    train_instance_segmentation = True
    run_training(checkpoint_name, model_type, train_instance_segmentation)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
