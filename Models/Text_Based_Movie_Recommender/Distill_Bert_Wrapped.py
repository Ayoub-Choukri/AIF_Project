import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer



class DistilBERTCLSExtractor(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    def forward(self, texts):
        # Tokenisation
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids']
        # Move to device
        input_ids = input_ids.to(self.device)
        
        attention_mask = encoded['attention_mask']

        # Passage dans le modèle
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # On récupère la première position ([CLS]-like)
        cls_embeddings = last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        return cls_embeddings
    


# import torch
# import torch.nn as nn
# from transformers import DistilBertModel, DistilBertTokenizer

class DistilBERTSentenceEmbedder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.to(self.device)  # Important : déplacer le modèle lui-même sur le bon device

    def forward(self, texts):
        # Tokenisation
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Passage dans le modèle
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Moyenne pondérée par le mask (pour ignorer les paddings)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)  # éviter division par zéro
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings  # (batch_size, hidden_dim)
