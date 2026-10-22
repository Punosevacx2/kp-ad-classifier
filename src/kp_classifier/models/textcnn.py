import torch
import torch.nn as nn
import torch.nn.functional as F

from kp_classifier.data.tokenizer import PAD_ID, UNK_ID


class ConvBranch(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, embedding_dropout, pooling="max"):
        super().__init__()
        if pooling not in ("max", "max_mean"):
            raise ValueError(f"Nepoznat pooling: {pooling}")
        self.pooling = pooling
        self.kernel_sizes = list(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, num_filters, k) for k in self.kernel_sizes)
        self.output_dim = num_filters * len(self.kernel_sizes) * (2 if pooling == "max_mean" else 1)

    def forward(self, ids, return_positions=False):
        if ids.size(1) < max(self.kernel_sizes):
            ids = F.pad(ids, (0, max(self.kernel_sizes) - ids.size(1)), value=PAD_ID)
        mask = ids != PAD_ID
        x = self.embedding_dropout(self.embedding(ids)).transpose(1, 2)
        pooled, positions = [], []
        for conv in self.convs:
            h = F.relu(conv(x))
            valid = mask[:, : h.size(2)].unsqueeze(1)
            values, idx = h.masked_fill(~valid, float("-inf")).max(dim=2)
            values = torch.where(torch.isinf(values), torch.zeros_like(values), values)
            pooled.append(values)
            positions.append(idx)
            if self.pooling == "max_mean":
                counts = valid.sum(dim=2).clamp(min=1)
                pooled.append((h * valid).sum(dim=2) / counts)
        features = torch.cat(pooled, dim=1)
        if return_positions:
            return features, positions
        return features


class HybridTextCNN(nn.Module):
    def __init__(
        self,
        word_vocab_size,
        char_vocab_size,
        num_classes,
        use_words=True,
        use_chars=True,
        word_embedding_dim=128,
        word_kernel_sizes=(1, 2, 3),
        word_filters=128,
        char_embedding_dim=32,
        char_kernel_sizes=(3, 4, 5),
        char_filters=128,
        embedding_dropout=0.1,
        word_dropout=0.1,
        hidden_dim=256,
        dropout=0.5,
        hidden_dropout=0.3,
        pooling="max",
    ):
        super().__init__()
        if not (use_words or use_chars):
            raise ValueError("Model mora koristiti bar jednu granu")
        self.use_words = use_words
        self.use_chars = use_chars
        self.word_dropout = word_dropout

        feature_dim = 0
        if use_words:
            self.word_branch = ConvBranch(
                word_vocab_size, word_embedding_dim, word_kernel_sizes, word_filters, embedding_dropout, pooling
            )
            feature_dim += self.word_branch.output_dim
        if use_chars:
            self.char_branch = ConvBranch(
                char_vocab_size, char_embedding_dim, char_kernel_sizes, char_filters, embedding_dropout, pooling
            )
            feature_dim += self.char_branch.output_dim
        self.feature_dim = feature_dim

        layers = [nn.Dropout(dropout)]
        if hidden_dim:
            layers += [nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(hidden_dropout)]
            layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            layers.append(nn.Linear(feature_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def apply_word_dropout(self, words):
        if not self.training or self.word_dropout <= 0:
            return words
        drop = (torch.rand(words.shape, device=words.device) < self.word_dropout) & (words != PAD_ID)
        return words.masked_fill(drop, UNK_ID)

    def features(self, words, chars):
        parts = []
        if self.use_words:
            parts.append(self.word_branch(self.apply_word_dropout(words)))
        if self.use_chars:
            parts.append(self.char_branch(chars))
        return torch.cat(parts, dim=1)

    def forward(self, words, chars):
        return self.head(self.features(words, chars))


MODEL_KEYS = [
    "use_words", "use_chars", "word_embedding_dim", "word_kernel_sizes", "word_filters",
    "char_embedding_dim", "char_kernel_sizes", "char_filters", "embedding_dropout",
    "word_dropout", "hidden_dim", "dropout", "hidden_dropout", "pooling",
]


def build_model(model_config, word_vocab_size, char_vocab_size, num_classes):
    kwargs = {k: model_config[k] for k in MODEL_KEYS if k in model_config}
    return HybridTextCNN(word_vocab_size, char_vocab_size, num_classes, **kwargs)


def parameter_table(model):
    rows = []
    for name, module in model.named_children():
        if name == "head":
            for i, layer in enumerate(module):
                n = sum(p.numel() for p in layer.parameters())
                if n:
                    rows.append((f"head.{i} {layer.__class__.__name__}", n))
            continue
        for sub_name, sub in module.named_children():
            n = sum(p.numel() for p in sub.parameters())
            if n:
                rows.append((f"{name}.{sub_name}", n))
    rows.append(("UKUPNO", sum(p.numel() for p in model.parameters())))
    return rows
