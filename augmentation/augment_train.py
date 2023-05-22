import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from augment_dataloader import AugmentDataset
from modules.utils import config_parser
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

if __name__ == "__main__":
    config = config_parser()

    model = AutoModelForMaskedLM.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = AugmentDataset(
        config=config,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=config["DataCollatorForLanguageModeling"]["mlm"],
        mlm_probability=config["DataCollatorForLanguageModeling"]["mlm_probability"],
    )

    training_args = TrainingArguments(
        output_dir=config["training_arguments"]["output_dir"],
        overwrite_output_dir=config["training_arguments"]["overwrite_output_dir"],
        learning_rate=config["training_arguments"]["learning_rate"],
        num_train_epochs=config["training_arguments"]["num_train_epochs"],
        per_gpu_train_batch_size=config["training_arguments"][
            "per_gpu_train_batch_size"
        ],
        save_steps=config["training_arguments"]["save_steps"],
        save_total_limit=config["training_arguments"]["save_total_limit"],
        logging_steps=config["training_arguments"]["logging_steps"],
        weight_decay=config["training_arguments"]["weight_decay"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(config["data_path"]["save_path"])
