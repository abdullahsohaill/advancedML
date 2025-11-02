# models_with_hints.py (Corrected and More Robust)

import torch
import torch.nn as nn

class VGGWithHint(nn.Module):
    """
    A generic wrapper for VGG-style models to extract intermediate hint layers.
    This wrapper preserves the original module names and handles architectural differences.
    """
    def __init__(self, base_model, hint_layer_index):
        super(VGGWithHint, self).__init__()
        self.features = base_model.features
        self.classifier = base_model.classifier
        self.hint_layer_index = hint_layer_index
        
        # --- THIS IS THE FIX ---
        # Check if the base model has an avgpool layer. If not, self.avgpool will be None.
        self.avgpool = getattr(base_model, 'avgpool', None)

    def forward(self, x):
        hint_features = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.hint_layer_index:
                hint_features = x
        
        # If the model has a separate avgpool layer, use it.
        if self.avgpool:
            x = self.avgpool(x)
        
        # Flatten and pass to classifier
        out = x.view(x.size(0), -1)
        logits = self.classifier(out)
        
        return logits, hint_features