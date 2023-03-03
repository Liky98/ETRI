from transformers import AutoModelForSequenceClassification,AutoTokenizer, AutoConfig, AutoModel
import torch.nn.functional as F
from torch import nn
import torch

class TextModel(nn.Module):
    def __init__(self, model_path, num_labels, device):
        super(TextModel, self).__init__()

        self.config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, num_labels=num_labels)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)

        self.hidden_size = self.config.hidden_size

        self.W_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.W_2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.W_3 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.W_4 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.device = device

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        size = len(input_ids[0]) - 1
        lower_triangular_indices = torch.tril_indices(size, size, offset=-1)
        start_indexs = lower_triangular_indices[0].to(self.device)
        end_indexs = lower_triangular_indices[1].to(self.device)
        first_token = output.last_hidden_state[:, 0, :]
        hidden_states = output.last_hidden_state[:, 1:, :]  # exclude first token

        W1_h = self.W_1(hidden_states)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = W1_h[:, start_indexs, :]
        W2_hj_emb = W2_h[:, end_indexs, :]
        W3_hi_start_emb = W3_h[:, start_indexs, :]
        W3_hi_end_emb = W3_h[:, end_indexs, :]
        W4_hj_start_emb = W4_h[:, start_indexs, :]
        W4_hj_end_emb = W4_h[:, end_indexs, :]

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        all_token = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + (W4_hj_start_emb * W4_hj_end_emb)
        all_token = torch.tanh(all_token)
        all_token = torch.mean(all_token, dim=1)

        return first_token, all_token


if __name__ =="__main__":
    path = "microsoft/deberta-V3-small"
    model = TextModel(path, num_labels=7,device='cpu')
    tokenizer = AutoTokenizer.from_pretrained(path)
    sample = "Hello World! I Want to run."
    input = tokenizer(sample, return_tensors="pt")

    input_ids = input.input_ids
    token_type_ids = input.token_type_ids
    attention_mask  = input.attention_mask

    logits = model(input_ids,token_type_ids,attention_mask)


