# models_with_embeddings.py

import torch.nn as nn

class VGGWithEmbedding(nn.Module):
    """
    A generic wrapper for VGG-style models to extract the pre-classifier embedding.
    """
    def __init__(self, base_model):
        super(VGGWithEmbedding, self).__init__()
        self.features = base_model.features
        self.classifier = base_model.classifier
        self.avgpool = getattr(base_model, 'avgpool', None)

    def forward(self, x):
        out = self.features(x)
        if self.avgpool:
            out = self.avgpool(out)
        
        # Flatten the features to get the embedding
        embedding = out.view(out.size(0), -1)
        
        # Pass the embedding through the classifier to get logits
        logits = self.classifier(embedding)
        
        return logits, embedding