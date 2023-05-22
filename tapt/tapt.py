import os
import sys

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset import TaptDataSet
from modules.utils import config_parser


def main():
    config = config_parser()

    MODEL_NAME = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    dataset = TaptDataSet(
        [config["path"]["train"], config["path"]["test"]], tokenizer=tokenizer
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./{MODEL_NAME}_TAPT",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=5e-6,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(f"./{MODEL_NAME}_TAPT_output")


if __name__ == "__main__":
    main()
