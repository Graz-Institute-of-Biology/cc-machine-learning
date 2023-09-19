from transformers import SegformerImageProcessor
from SemSegDataset import SemanticSegmentationDataset
from torch.utils.data import DataLoader
from tuner import SegformerFinetuner
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os

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

trainer.fit(segformer_finetuner)