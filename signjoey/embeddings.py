import math
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from signjoey.helpers import freeze_params

def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))


class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])


# TODO (Cihan): Spatial and Word Embeddings are pretty much the same
#       We might as well convert them into a single module class.
#       Only difference is the lut vs linear layers.
class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 8,
        scale: bool = False,
        scale_factor: float = None,
        norm_type: str = None,
        activation_type: str = None,
        vocab_size: int = 0,
        padding_idx: int = 1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param mask: token masks
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """

        x = self.lut(x)


        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )


class SpatialEmbeddings(nn.Module):

    """
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        scale: bool = False,
        scale_factor: float = None,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.
        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        # self.ln = nn.Linear(self.input_size, self.embedding_dim) #@jinhui
        self.ln = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim))

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """

        x = self.ln(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, input_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_size,
        )

# jinhui

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
import torch.nn as nn

class TextEmbeddings(nn.Module):

    """
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        scale: bool = False,
        scale_factor: float = None,
        max_length = 256,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """

        bert_path = "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/checkpoints/bert-base-german-cased"
        super().__init__()

        bert = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels=embedding_dim)
        word_embedded = bert.base_model.embeddings
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.bert_word_embedding = word_embedded.word_embeddings # Embeddings
        self.vocab_size = self.bert_word_embedding.num_embeddings
        self.input_size = self.bert_word_embedding.embedding_dim
        self.embedding_dim = embedding_dim
        self.ln = nn.Linear(self.input_size, self.embedding_dim)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: [], mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """
        x = self.tokenize(x) # @jinhui 之后要放到外面取执行，并切to dievce
        x = self.bert_word_embedding(x)
        x = self.ln(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )

    def tokenize(self, raw_tokens):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.tokenizer.pad_token_id) #该用什么填充呢？
        indexed_tokens = indexed_tokens[:self.max_length]

        return torch.tensor(indexed_tokens, dtype=torch.long)

class GlossEmbeddings(nn.Module):

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 8,
        scale: bool = False,
        scale_factor: float = None,
        norm_type: str = None,
        activation_type: str = None,
        vocab_size: int = 0,
        padding_idx: int = 2,
        freeze: bool = False,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)
        self.feature_map = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.embedding_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim*2,self.embedding_dim))

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

        # pylint: disable=arguments-differ

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param mask: token masks
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """

        x = self.lut(x)

        x = self.feature_map(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )

    def _tokenize(self, raw_tokens):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.tokenizer.pad_token_id)
        indexed_tokens = indexed_tokens[:self.max_length]

        return torch.tensor(indexed_tokens, dtype=torch.long)
