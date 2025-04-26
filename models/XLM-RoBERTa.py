import torch
from torch import nn
from transformers import XLMRobertaModel


class SarcasmDetectorXLM(nn.Module):
    def __init__(self, pretrained_model_name="xlm-roberta-base"):
        super(SarcasmDetectorXLM, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name)

        hidden_size = 768

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),  # Binary classification (sarcastic or not)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[
            :, 0, :
        ]  # Get the [CLS] token embedding
        return self.classifier(pooled_output)

    @staticmethod
    def create_and_prepare_model(
        pretrained_model_name="xlm-roberta-base", device="cuda"
    ):
        model = SarcasmDetectorXLM(pretrained_model_name)

        # Freeze the lower layers of XLM-RoBERTa and only fine-tune the top layers
        for name, param in model.roberta.named_parameters():
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