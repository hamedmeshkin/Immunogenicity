import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, BertConfig, BertModel


# 1. Define tokenizer model
tokenizer = Tokenizer(models.BPE())

# 2. Set up pre-tokenizer: character-level (since amino acids are single letters)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. Train from FASTA or text corpus (amino acid sequences)
trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]"])
files = ["Heavy_Light.Data/light_chains.fasta", "Heavy_Light.Data/heavy_chains.fasta"] # One sequence per line

tokenizer.train(files, trainer)

# 4. Save and test
tokenizer.save("antibody_tokenizer_non_trained.json")
#print(tokenizer.encode("EVQLVESGGGLIQP").tokens)



class AntiBERTy_2(nn.Module):
    def __init__(self):
        super(AntiBERTy_2, self).__init__()

        # Load your tokenizer
        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="antibody_tokenizer_non_trained.json")
        self.fast_tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})

        # Set up configuration: 8 layers, 8 heads, hidden size 512
        # Train a BERT model with this tokenizer
        config = BertConfig(
            vocab_size=self.fast_tokenizer.vocab_size,
            hidden_size=512,
            intermediate_size=2048,
            max_position_embeddings=512,
            num_hidden_layers=8,
            num_attention_heads=8,

        )

        self.bert = BertModel(config)

    def forward(self, sequence, attention_mask=None):
        """
        input_ids: Tensor of shape [B, L] (batch size, sequence length)
        attention_mask: Optional mask for padding (same shape)
        """
        input_ids,attention_mask = self.seq2num(sequence)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get per-residue (token) embeddings
        sequence_embeddings = output.last_hidden_state  # Shape: [B, L, 512]

        return sequence_embeddings.detach()

    def seq2num(self,sequences,padding=True,truncation=True,return_attention_mask=True):

        inputs = self.fast_tokenizer(sequences, return_tensors="pt", padding=padding, truncation=truncation,return_attention_mask=return_attention_mask)

        if return_attention_mask:
            return inputs["input_ids"], inputs["attention_mask"]
        else:
            return inputs["input_ids"]



# How this code works (Example)
'''
# Shape: [1, L]
model = AntiBERTy_2()
embeddings  = model(sequence)  # Shape: [1, 150, 512]
per_residue = embeddings.squeeze(0)    # Shape: [150, 512] = L x 512

sequences = Model.train_dataset[['Heavy_Chain', 'Light_Chain']]
embeddings = [model(list(sequences.iloc[i])) for i in range(len(sequences))]
'''

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
# Step1
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
)

tokenizer.train(["Heavy_Light.Data/light_chains.fasta", "Heavy_Light.Data/heavy_chains.fasta"], trainer)
tokenizer.save("antibody_tokenizer.json")


# Step 2
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="antibody_tokenizer.json")
hf_tokenizer.save_pretrained("AntibodyTokenizer")
hf_tokenizer.add_special_tokens({
    "pad_token": "[PAD]",
    "mask_token": "[MASK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "unk_token": "[UNK]"
})



config = BertConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=512,
    num_attention_heads=8,
    num_hidden_layers=8,
    intermediate_size=2048,
    max_position_embeddings=512
)

model = BertForMaskedLM(config)
model.resize_token_embeddings(len(hf_tokenizer))
model.save_pretrained("AntiBERTyModel")

# Step 3
dataset = load_dataset("text", data_files={"train": ["Heavy_Light.Data/light_chains.fasta", "Heavy_Light.Data/heavy_chains.fasta"]})

# Tokenize
def tokenize_function(examples):
    return hf_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# MLM Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./AntiBERTyLM",
    per_device_train_batch_size=32,
#    evaluation_strategy="no",
    num_train_epochs=4,
    save_steps=10000,
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=hf_tokenizer,
    data_collator=data_collator,
)

trainer.train()# resume_from_checkpoint="./AntiBERTyLM/checkpoint-384000")
#
#
trainer.save_model("trained_antiberty")
hf_tokenizer.save_pretrained("trained_antiberty")#
##
# ###################################################################
