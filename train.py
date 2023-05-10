import os
from datetime import datetime

import pytorch_lightning as pl
import pytz
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
from model_list import CustomBertForSequenceClassification
from dataloader import ERDataModule
from model import ERNet

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    prj_dir = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_NAME = "klue/bert-base"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[ORG]", "[PER]", "[LOC]", "[POH]",
                                                                                  "[DAT]", "[NOH]", "[/ORG]", "[/PER]",
                                                                                  "[/LOC]", "[/POH]", "[/DAT]",
                                                                                  "[/NOH]",'[SUB]','[/SUB]','[OBJ]','[/OBJ]']})
    dataset_train_dir = os.path.join(prj_dir , "dataset", "data", "train.csv")
    dataset_val_dir = os.path.join(prj_dir, "dataset", "data", "val_.csv")
    dataset_test_dir = os.path.join(prj_dir, "dataset", "data", "test.csv")
    dataloader = ERDataModule(dataset_train_dir=dataset_train_dir,dataset_val_dir=dataset_val_dir,dataset_test_dir=dataset_test_dir, tokenizer=tokenizer, batch_size=16)
    model = ERNet(model_name=MODEL_NAME, learning_rate=0.00005, weight_decay=1e-4,state='train')
    model.model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    now = datetime.now(pytz.timezone("Asia/Seoul"))
    print(model)
    trainer = pl.Trainer(callbacks=ModelCheckpoint(dirpath=f"./checkpoint/{MODEL_NAME.replace('/', '_')}/{now.strftime('%Y-%m-%d_%H_%M_%S')}/", filename="{epoch}-{val_micro_f1:.2f}", monitor="val_micro_f1", mode="max"), max_epochs = 6)
    trainer.fit(model = model, train_dataloaders=dataloader)

