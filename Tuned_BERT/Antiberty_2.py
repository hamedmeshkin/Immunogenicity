import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, BertConfig, BertModel



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
