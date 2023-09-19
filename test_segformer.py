from argparse import ArgumentParser
from PIL import Image
# from mmseg.apis import inference_model, init_model, show_result_pyplot
# from mmseg.utils import get_palette
from transformers import SegformerImageProcessor
from SemSegDataset import SemanticSegmentationDataset
from torch.utils.data import DataLoader
from tuner import SegformerFinetuner
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, SegformerForSemanticSegmentation , SegformerConfig
import torch
from torch import nn
# from torchviz import make_dot

def test():

    checkpoint = "/usr/people/EDVZ/faulhamm/cc-machine-learning/lightning_logs/version_8/checkpoints/epoch=0-step=108.ckpt"
    base_path = "/usr/people/EDVZ/faulhamm/cc-machine-learning/test"
    img_paths = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".JPG")]
    config = "config.py"

    print(img_paths)

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint=checkpoint, device="gpu")
    # test a single image
    results = inference_model(model, img_paths)
    print(results[0])
    # show the results
    # show_result_pyplot(model, args.img, result, get_palette(args.palette))


def test2():
    checkpoint = "/usr/people/EDVZ/faulhamm/cc-machine-learning/lightning_logs/version_10/checkpoints/last.ckpt"

    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 128

    train_dataset = SemanticSegmentationDataset("train/", feature_extractor)
    val_dataset = SemanticSegmentationDataset("valid/", feature_extractor)
    test_dataset = SemanticSegmentationDataset("test/", feature_extractor)

    print(train_dataset)

    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=8)


    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        test_dataloader=test_dataloader, 
        metrics_interval=10,
    )
    # segformer_finetuner = segformer_finetuner.load_from_checkpoint(checkpoint)
    print(segformer_finetuner.id2label)


    early_stop_callback = EarlyStopping(
        monitor="mean_iou", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=0, monitor="epoch",every_n_epochs=10, save_last=True)

    trainer = pl.Trainer(
        # gpus=1, 
        callbacks=[checkpoint_callback],
        max_epochs=1000,
        val_check_interval=len(train_dataloader),
        logger=True,
    )

    res = trainer.predict(segformer_finetuner, test_dataloader)
    print(res)

def test3():
    img = Image.open("/usr/people/EDVZ/faulhamm/cc-machine-learning/test/150223_TF_M_S_DJI_0673_part_2.JPG")

    checkpoint = torch.load("/usr/people/EDVZ/faulhamm/cc-machine-learning/lightning_logs/version_8/checkpoints/epoch=0-step=108.ckpt")
    state_dict = checkpoint["state_dict"]
    configuration = SegformerConfig()
    print(configuration)
    model = SegformerForSemanticSegmentation(configuration) 
    model.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", state_dict=state_dict)
    model.eval()

    with torch.no_grad():
        output = model(img)
    
    print(output.shape)

def test4():
    checkpoint = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\last.ckpt"
    # checkpoint = "/usr/people/EDVZ/faulhamm/cc-machine-learning/lightning_logs/version_8/checkpoints/epoch=0-step=108.ckpt"
    checkpoint = torch.load(checkpoint)
    print(checkpoint["state_dict"].keys())
    make_dot(checkpoint["state_dict"]).render("graph", format="png")


if __name__ == "__main__":
    test3()