# Step	Purpose	                        Key Function/Module

# 1	    Install dependencies	        pip
# 2	    Load environment variables	    dotenv
# 3	    Authenticate	                huggingface_hub.login
# 4	    Load model/tokenizer	        transformers.AutoModelForCausalLM, AutoTokenizer
# 5	    Test generation	                model.generate
# 6	    Load dataset	                datasets.load_dataset
# 7	    Tokenize data	                dataset.map
# 8	    Data collator	                DataCollatorForLanguageModeling
# 9	    Apply LoRA	                    peft.get_peft_model, LoraConfig
# 10	Training arguments	            TrainingArguments
# 11	Trainer setup/train	            Trainer
# 12	Save model/tokenizer	        trainer.save_model, tokenizer.save_pretrained
# 13	Reload and use	                AutoModelForCausalLM.from_pretrained




#0. Install Required Libraries
# !pip install datasets pandas torch transformers[torch] python-dotenv peft

# =========================
# Hugging Face Model Fine-Tuning Pipeline (with LoRA)
# =========================

# 1. Imports
import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

# 2. Dynamic Variables (Change as needed)
MODEL_NAME = "distilgpt2"
DATASET_NAME = "tniranjan/aitamilnadu_tamil_stories_no_instruct"
DATASET_SPLIT = "train[:1000]"
TRAIN_SIZE = 0.95
MAX_LENGTH = 200
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
SAVE_STEPS = 500
LOGGING_STEPS = 50

BASE_MODEL_DIR = os.path.join("HFModels", MODEL_NAME)
BASE_DATASET_DIR = os.path.join("HFDataset", DATASET_NAME.split('/')[-1])
BASE_FINETUNED_DIR = os.path.join("HFFinetunedModel", f"fine_tuned_{MODEL_NAME}_Tamil")

# 3. Load Environment Variables and Authenticate
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# 4. Load Pretrained Model and Tokenizer
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.save_pretrained(BASE_MODEL_DIR)
tokenizer.save_pretrained(BASE_MODEL_DIR)

# 5. Test Model Generation (Optional)
test_text = "ஒரு நாள் "
inputs = tokenizer(test_text, return_tensors="pt")
output = model.generate(inputs.input_ids, max_new_tokens=100)
print("Sample output from base model:\n", tokenizer.decode(output[0], skip_special_tokens=True))

# 6. Load and Prepare Dataset
os.makedirs(BASE_DATASET_DIR, exist_ok=True)
local_dataset_file = os.path.join(BASE_DATASET_DIR, "train-1000.arrow")
if os.path.exists(local_dataset_file):
    print(f"Loading dataset from local file: {local_dataset_file}")
    raw_data = load_from_disk(BASE_DATASET_DIR)
else:
    print(f"Downloading dataset: {DATASET_NAME}")
    raw_data = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    raw_data.save_to_disk(BASE_DATASET_DIR)
data = raw_data.train_test_split(train_size=TRAIN_SIZE)

# 7. Tokenize Dataset
def preprocess_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=MAX_LENGTH)
tokenized_dataset = data.map(
    preprocess_batch,
    batched=True,
    batch_size=4,
    remove_columns=data["train"].column_names
)
print("Tokenized dataset preview:", tokenized_dataset)

# 8. Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 9. Apply LoRA (Parameter-Efficient Fine-Tuning)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()  # Set model to training mode

# 10. Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=LOGGING_STEPS,
    logging_dir="./logs",
    resume_from_checkpoint=True
)

# 11. Initialize Trainer and Start Training
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    optimizers=(optimizer, None),
    data_collator=data_collator
)
trainer.train()

# 12. Save the Fine-Tuned Model and Tokenizer Locally
os.makedirs(BASE_FINETUNED_DIR, exist_ok=True)
trainer.save_model(BASE_FINETUNED_DIR)
tokenizer.save_pretrained(BASE_FINETUNED_DIR)

# 13. Load and Use the Fine-Tuned Model
tokenizer = AutoTokenizer.from_pretrained(BASE_FINETUNED_DIR)
model = AutoModelForCausalLM.from_pretrained(BASE_FINETUNED_DIR)
test_text = "ஒரு நாள்"
inputs = tokenizer(test_text, return_tensors="pt")
output = model.generate(inputs.input_ids, max_new_tokens=100)
print("Sample output from fine-tuned model:\n", tokenizer.decode(output[0], skip_special_tokens=True))

# 14. (Optional) Fine-tune Again and Save with Incremented Version
version = 1
while os.path.exists(f"{BASE_FINETUNED_DIR}_v{version}"):
    version += 1
finetuned_dir_versioned = f"{BASE_FINETUNED_DIR}_v{version}"
os.makedirs(finetuned_dir_versioned, exist_ok=True)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    optimizers=(optimizer, None),
    data_collator=data_collator
)
trainer.train()
trainer.save_model(finetuned_dir_versioned)
tokenizer.save_pretrained(finetuned_dir_versioned)

# =========================
# End of Pipeline
# =========================






#---------------------------------------------------------------------------------------------------------------------------------------


# # =========================
# # Hugging Face Model Fine-Tuning Pipeline (with LoRA)
# # Modular Version: Each step as a function
# # =========================

# import os
# from dotenv import load_dotenv
# from huggingface_hub import login
# from datasets import load_dataset, load_from_disk
# from transformers import (
#     AutoModelForCausalLM, AutoTokenizer,
#     DataCollatorForLanguageModeling,
#     TrainingArguments, Trainer
# )
# from peft import get_peft_model, LoraConfig, TaskType
# from torch.optim import AdamW

# # --------- Dynamic Variables (Change as needed) ---------
# MODEL_NAME = "distilgpt2"
# DATASET_NAME = "tniranjan/aitamilnadu_tamil_stories_no_instruct"
# DATASET_SPLIT = "train[:1000]"
# TRAIN_SIZE = 0.95
# MAX_LENGTH = 200
# BATCH_SIZE = 2
# NUM_EPOCHS = 3
# LEARNING_RATE = 1e-5
# WEIGHT_DECAY = 0.01
# SAVE_STEPS = 500
# LOGGING_STEPS = 50

# BASE_MODEL_DIR = os.path.join("HFModels", MODEL_NAME)
# BASE_DATASET_DIR = os.path.join("HFDataset", DATASET_NAME.split('/')[-1])
# BASE_FINETUNED_DIR = os.path.join("HFFinetunedModel", f"fine_tuned_{MODEL_NAME}_Tamil")

# # --------- Step 1: Load Environment and Authenticate ---------
# def authenticate_hf():
#     load_dotenv()
#     hf_token = os.getenv("HF_TOKEN")
#     login(token=hf_token)

# # --------- Step 2: Load Model and Tokenizer ---------
# def load_model_and_tokenizer(model_name, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.save_pretrained(save_dir)
#     tokenizer.save_pretrained(save_dir)
#     return model, tokenizer

# # --------- Step 3: Test Model Generation (Optional) ---------
# def test_generation(model, tokenizer, prompt, max_new_tokens=100, label="base model"):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     output = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
#     print(f"Sample output from {label}:\n", tokenizer.decode(output[0], skip_special_tokens=True))

# # --------- Step 4: Load and Prepare Dataset ---------
# def load_and_prepare_dataset(dataset_name, dataset_split, local_dir, train_size):
#     os.makedirs(local_dir, exist_ok=True)
#     local_dataset_file = os.path.join(local_dir, "train-1000.arrow")
#     if os.path.exists(local_dataset_file):
#         print(f"Loading dataset from local file: {local_dataset_file}")
#         raw_data = load_from_disk(local_dir)
#     else:
#         print(f"Downloading dataset: {dataset_name}")
#         raw_data = load_dataset(dataset_name, split=dataset_split)
#         raw_data.save_to_disk(local_dir)
#     data = raw_data.train_test_split(train_size=train_size)
#     return data

# # --------- Step 5: Tokenize Dataset ---------
# def tokenize_dataset(data, tokenizer, max_length):
#     def preprocess_batch(batch):
#         return tokenizer(batch["text"], truncation=True, padding=True, max_length=max_length)
#     tokenized_dataset = data.map(
#         preprocess_batch,
#         batched=True,
#         batch_size=4,
#         remove_columns=data["train"].column_names
#     )
#     print("Tokenized dataset preview:", tokenized_dataset)
#     return tokenized_dataset

# # --------- Step 6: Data Collator ---------
# def get_data_collator(tokenizer):
#     return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # --------- Step 7: Apply LoRA ---------
# def apply_lora(model):
#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.1,
#         bias="none",
#         task_type=TaskType.CAUSAL_LM
#     )
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
#     model.train()
#     return model

# # --------- Step 8: Training Arguments ---------
# def get_training_args(save_steps, learning_rate, weight_decay, num_epochs, batch_size, logging_steps):
#     return TrainingArguments(
#         output_dir="./output",
#         evaluation_strategy="epoch",
#         save_steps=save_steps,
#         learning_rate=learning_rate,
#         weight_decay=weight_decay,
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         logging_steps=logging_steps,
#         logging_dir="./logs",
#         resume_from_checkpoint=True
#     )

# # --------- Step 9: Trainer and Training ---------
# def train_model(model, tokenized_dataset, training_args, data_collator, learning_rate):
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     trainer = Trainer(
#         model=model,
#         train_dataset=tokenized_dataset["train"],
#         eval_dataset=tokenized_dataset["test"],
#         args=training_args,
#         optimizers=(optimizer, None),
#         data_collator=data_collator
#     )
#     trainer.train()
#     return trainer

# # --------- Step 10: Save Model and Tokenizer ---------
# def save_model_and_tokenizer(trainer, tokenizer, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     trainer.save_model(save_dir)
#     tokenizer.save_pretrained(save_dir)

# # --------- Step 11: Reload and Test Fine-Tuned Model ---------
# def reload_and_test_finetuned(save_dir, prompt):
#     tokenizer = AutoTokenizer.from_pretrained(save_dir)
#     model = AutoModelForCausalLM.from_pretrained(save_dir)
#     test_generation(model, tokenizer, prompt, label="fine-tuned model")
#     return model, tokenizer

# # --------- Step 12: Optional - Fine-tune Again and Save Versioned ---------
# def finetune_and_save_versioned(model, tokenizer, tokenized_dataset, training_args, data_collator, learning_rate, base_dir):
#     version = 1
#     while os.path.exists(f"{base_dir}_v{version}"):
#         version += 1
#     finetuned_dir_versioned = f"{base_dir}_v{version}"
#     os.makedirs(finetuned_dir_versioned, exist_ok=True)
#     model.train()
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     trainer = Trainer(
#         model=model,
#         train_dataset=tokenized_dataset["train"],
#         eval_dataset=tokenized_dataset["test"],
#         args=training_args,
#         optimizers=(optimizer, None),
#         data_collator=data_collator
#     )
#     trainer.train()
#     trainer.save_model(finetuned_dir_versioned)
#     tokenizer.save_pretrained(finetuned_dir_versioned)
#     print(f"Saved versioned fine-tuned model at: {finetuned_dir_versioned}")

# # =========================
# # Main Pipeline
# # =========================
# if __name__ == "__main__":
#     # Step 1: Authenticate
#     authenticate_hf()

#     # Step 2: Load Model and Tokenizer
#     model, tokenizer = load_model_and_tokenizer(MODEL_NAME, BASE_MODEL_DIR)

#     # Step 3: Test Base Model Generation
#     test_generation(model, tokenizer, "ஒரு நாள் ", label="base model")

#     # Step 4: Load and Prepare Dataset
#     data = load_and_prepare_dataset(DATASET_NAME, DATASET_SPLIT, BASE_DATASET_DIR, TRAIN_SIZE)

#     # Step 5: Tokenize Dataset
#     tokenized_dataset = tokenize_dataset(data, tokenizer, MAX_LENGTH)

#     # Step 6: Data Collator
#     data_collator = get_data_collator(tokenizer)

#     # Step 7: Apply LoRA
#     model = apply_lora(model)

#     # Step 8: Training Arguments
#     training_args = get_training_args(SAVE_STEPS, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE, LOGGING_STEPS)

#     # Step 9: Trainer and Training
#     trainer = train_model(model, tokenized_dataset, training_args, data_collator, LEARNING_RATE)

#     # Step 10: Save Fine-Tuned Model and Tokenizer
#     save_model_and_tokenizer(trainer, tokenizer, BASE_FINETUNED_DIR)

#     # Step 11: Reload and Test Fine-Tuned Model
#     model, tokenizer = reload_and_test_finetuned(BASE_FINETUNED_DIR, "ஒரு நாள்")

#     # Step 12: (Optional) Fine-tune Again and Save Versioned
#     finetune_and_save_versioned(model, tokenizer, tokenized_dataset, training_args, data_collator, LEARNING_RATE, BASE_FINETUNED_DIR)

# # =========================
# # End of Modular Pipeline
# # =========================
