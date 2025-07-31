import os
import torch
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
from ast import literal_eval
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelFineTuner:
    def __init__(self, config):
        self.model_name = config['MODEL_NAME']
        self.output_dir = config['OUTPUT_DIR']
        self.hf_token = config['HF_TOKEN']
        self.r = config['LORA_R']
        self.max_seq_length = config['MAX_SEQ_LENGTH']
        self.batch_size = config['BATCH_SIZE']
        self.epochs = config['NUM_EPOCHS']
        self.learning_rate = config['LEARNING_RATE']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def login_huggingface(self):
        from huggingface_hub import login
        login(token=self.hf_token, write_permission=True)

    def process_data(self, df, input_column, target_column):
        def format_prompt(example):
            return f"""<s>[INST] You are a careful data annotator. Your task is to assign up to five unique and relevant tags to the provided text without explanations. Return the tags in the format: ['Tag1','Tag2',...,'Tag5']. The text is: "{example[input_column]}". The tags must be in the same language as the text. [/INST] {example[target_column]} </s>"""
        
        df = df.assign(text=lambda x: x[[input_column, target_column]].apply(
            lambda row: format_prompt(row), axis=1))
        return df

    def load_data(self, file_path, input_column, target_column):
        df = pd.read_csv(file_path)
        df = df[[input_column, target_column]].rename(columns={target_column: 'tags'})
        df['tags'] = df['tags'].apply(literal_eval)
        df = df[df['tags'].apply(len) >= 3]
        train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
        train_df = self.process_data(train_df, input_column, target_column)
        return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = GPTQConfig(bits=4, disable_exllama=True, tokenizer=self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        if torch.cuda.device_count() > 1:
            self.model.is_parallelizable = True
            self.model.model_parallel = True
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        peft_config = LoraConfig(
            r=self.r, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

    def train(self, train_dataset):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=5,
            num_train_epochs=self.epochs,
            fp16=True,
            push_to_hub=True,
            hub_private_repo=True
        )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            peft_config=None,
            dataset_text_field="text",
            args=training_args,
            tokenizer=self.tokenizer,
            packing=False,
            max_seq_length=self.max_seq_length
        )
        trainer.train()
        trainer.push_to_hub()

def main():
    config = {
        'MODEL_NAME': os.getenv('MODEL_NAME'),
        'OUTPUT_DIR': os.getenv('OUTPUT_DIR'),
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'LORA_R': int(os.getenv('LORA_R')),
        'MAX_SEQ_LENGTH': int(os.getenv('MAX_SEQ_LENGTH')),
        'BATCH_SIZE': int(os.getenv('BATCH_SIZE')),
        'NUM_EPOCHS': int(os.getenv('NUM_EPOCHS')),
        'LEARNING_RATE': float(os.getenv('LEARNING_RATE')),
        'DATA_PATH': os.getenv('DATA_PATH'),
        'INPUT_COLUMN': os.getenv('INPUT_COLUMN'),
        'TARGET_COLUMN': os.getenv('TARGET_COLUMN')
    }
    
    fine_tuner = ModelFineTuner(config)
    fine_tuner.login_huggingface()
    train_dataset, _ = fine_tuner.load_data(
        config['DATA_PATH'], config['INPUT_COLUMN'], config['TARGET_COLUMN']
    )
    fine_tuner.setup_model()
    fine_tuner.train(train_dataset)

if __name__ == "__main__":
    main()