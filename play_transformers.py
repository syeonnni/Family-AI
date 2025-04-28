from torch import nn
import torch
import torch.nn.functional as F
from math import sqrt

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def scaled_dot_product_attention(self, query, key, value):
        dim_k = key.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)

    def forward(self, hidden_state):
        return self.scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        return self.output_linear(x)


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # 입력 시퀀스에 대해 위치 ID를 만듭니다.
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

        # 위치 ID를 GPU로 이동
        position_ids = position_ids.to(input_ids.device)

        # 토큰 임베딩과 위치 임베딩을 만듭니다.
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # 토큰 임베딩과 위치 임베딩을 합칩니다.
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TrasformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TrasformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        # skip connection
        x = x + self.attention(hidden_state)
        return self.feed_forward(self.layer_norm_2(x)) + x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            [
                TrasformerEncoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(TransformerForSequenceClassification, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        

        # GPU 사용 가능 여부 확인 및 device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device) # 모델을 GPU로 이동

    def forward(self, x):
        # 입력을 GPU로 이동
        x = x.to(self.device)
        x = self.encoder(x)[:, 0, :]
        x = self.dropout(x)
        x = self.classifier(x)
        return F.softmax(x, dim=-1)