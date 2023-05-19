from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from augment_dataloader import AugmentDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

if __name__ == '__main__':
    model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    dataset = AugmentDataset(["../dataset/test/test_data.csv", "../dataset/train/train.csv"], tokenizer=tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir='./pretrain2',
        overwrite_output_dir=True,
        learning_rate=1e-5,
        num_train_epochs=2,
        per_gpu_train_batch_size=16,
        #evaluation_strategy='epoch',
        #save_strategy ='epoch',
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        weight_decay=0.01,
        #load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset   
    )

    trainer.train()
    trainer.save_model("./augmentation_model")