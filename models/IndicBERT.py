import torch
from torch import nn
from transformers import AutoModel


class IndicBERTSarcasmDetector(nn.Module):
    def __init__(self, pretrained_model_name="ai4bharat/indic-bert"):
        super(IndicBERTSarcasmDetector, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased dropout for better regularization
            nn.Linear(256, 2),  # Binary classification (sarcastic or not)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[
            :, 0, :
        ]  # Get the [CLS] token embedding
        return self.classifier(pooled_output)

    @staticmethod
    def create_and_prepare_model(
        pretrained_model_name="ai4bharat/indic-bert", device="cuda"
    ):
        base_model = AutoModel.from_pretrained(pretrained_model_name)
        model = IndicBERTSarcasmDetector(pretrained_model_name)

        # Freeze the lower layers of BERT and only fine-tune the top layers
        # This helps prevent overfitting on small datasets
        for name, param in model.bert.named_parameters():
            param.requires_grad = False  # Freeze all by default

            # Only unfreeze the top 3 layers and embeddings
            if (
                "encoder.layer.9" in name
                or "encoder.layer.10" in name
                or "encoder.layer.11" in name
                or "embeddings" in name
            ):
                param.requires_grad = True

        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return model