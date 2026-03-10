import torch
from torch import Tensor

from ltsa.image_encoder import ImageEncoder
from ltsa.tpe import TemporalPositionalEncoding
from ltsa.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


class LTSA(torch.nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        n_heads: int,
        dropout: float,
        n_layers: int,
        max_seq_len: int,
        n_classes: int,
        device: torch.device | None = None,
    ):
        super(LTSA, self).__init__()
        self.max_seq_len: int = max_seq_len
        self.device: torch.device | None = device

        self.encoder: ImageEncoder = image_encoder
        transformer_encoder = TransformerEncoderLayer(
            d_model=self.encoder.n_features,
            nhead=n_heads,
            dim_feedforward=self.encoder.n_features,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer: TransformerEncoder = TransformerEncoder(
            encoder_layer=transformer_encoder, num_layers=n_layers
        )

        self.pos_encoder: TemporalPositionalEncoding = TemporalPositionalEncoding(
            d_model=self.encoder.n_features, dropout=0, max_len=max_seq_len * 12
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=self.encoder.n_features, out_features=n_classes),
            torch.nn.Sigmoid(),
        )

        self.step_ahead_predictor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=self.encoder.n_features, out_features=self.encoder.n_features),
        )

        self.causal_mask: Tensor = torch.triu(
            input=torch.full(size=(max_seq_len, max_seq_len), fill_value=float("-inf"), device="cuda:0"),
            diagonal=1,
        )

    def forward(
        self, x, seq_lengths, rel_times, prior_AMD_sevs
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # embeddings: batch_size*max_seq_len x n_features
        embeddings: Tensor = self.encoder(x)

        # Reshape embeddings to batch_size x seq_length x n_features
        embeddings: Tensor = embeddings.reshape(len(seq_lengths), self.max_seq_len, self.encoder.n_features)

        x: Tensor = self.pos_encoder(embeddings, rel_times)

        # Create mask to ignore padding tokens. For each sequence of visits, mask all tokens beyond last visit
        # Here, 1 = pad (ignore), 0 = valid (keep)
        src_key_padding_mask: Tensor = (
            torch.ones(size=(x.shape[0], x.shape[1])).float().to(device=self.device)
        )  # batch x seq_length
        for i, seq_length in enumerate(seq_lengths):
            src_key_padding_mask[i, :seq_length] = 0

        # Transformer modeling with "decoder-style" causal attention (only attend to current + PRIOR elements of each sequence)
        feats, attn_map = self.transformer(
            x,
            mask=self.causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
            need_weights=True,
        )

        # Using src_key_padding_mask undoes padding... so re-pad each sequence in the batch with zeroes
        if feats.shape[1] < self.max_seq_len:
            feats: Tensor = torch.nn.functional.pad(
                input=feats, pad=(0, 0, 0, self.max_seq_len - feats.shape[1], 0, 0), mode="constant"
            )

        # Predict discrete-time hazard distribution
        hazards: Tensor = self.classifier(feats)

        # Generative discrete-time survival probabilities
        surv: Tensor = torch.cumprod(input=1 - hazards.view(-1, hazards.shape[-1]), dim=1).view(
            hazards.shape[0], hazards.shape[1], hazards.shape[2]
        )

        # Padding mask used to compute loss later
        padding_mask: Tensor = torch.bitwise_not(input=src_key_padding_mask.bool()).unsqueeze(dim=-1)

        # Get time elapsed (delta) between consecutive visits
        delta_times: Tensor = torch.diff(input=rel_times)  # batch x max_seq_len-1
        delta_times[delta_times < 0] = 0
        delta_times: Tensor = torch.nn.functional.pad(
            input=delta_times, pad=(0, 1), mode="constant", value=0
        )  # batch x max_seq_len

        # Use relative temporal timestep encoding to inform the model of "# months of into the future for which to predict imaging features"
        delta_encoded_feats: Tensor = self.pos_encoder(feats, delta_times)

        # Predict imaging features of *next* visit for each subsequence
        feat_preds: Tensor = self.step_ahead_predictor(delta_encoded_feats)

        # Get actual imaging features of next visit
        feat_targets: Tensor = torch.nn.functional.pad(
            input=feats[:, 1:, :], pad=(0, 0, 0, 1), mode="constant", value=0
        )

        return hazards, surv, feat_preds, feat_targets, padding_mask, attn_map


class ImageSurvivalModel(torch.nn.Module):
    def __init__(self, image_encoder: ImageEncoder, n_classes: int, dropout: float = 0.25):
        super(ImageSurvivalModel, self).__init__()

        self.encoder: ImageEncoder = image_encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=self.encoder.n_features, out_features=n_classes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x) -> tuple[Tensor, Tensor]:
        x: Tensor = self.encoder(x)

        hazards: Tensor = self.classifier(x)
        surv: Tensor = torch.cumprod(input=1 - hazards, dim=1)

        return hazards, surv
