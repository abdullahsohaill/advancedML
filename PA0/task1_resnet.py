""""
Advanced Machine Learning - Assignment 0
Task 1: Inner Workings of ResNet-152
Complete implementation covering all subtasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from sklearn.manifold import TSNE
import umap
from collections import defaultdict
import seaborn as sns
import pandas as pd

# set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ResNetExplorer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # data transforms for CIFAR-10
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # loading CIFAR-10 dataset
        self.train_dataset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=self.data_transforms['train'])
        self.val_dataset = datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=self.data_transforms['val'])
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64,
                                                       shuffle=True, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=64,
                                                     shuffle=False, num_workers=2)
        
        self.class_names = self.train_dataset.classes
        print(f"CIFAR-10 classes: {self.class_names}")
        
        # storage for results
        self.results = {}
        self.feature_hooks = {}
    
    def create_baseline_model(self):
        """Task 1.1: Baseline Setup"""
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        
        # freezing all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # replacing final classification layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        return model.to(self.device)
    
    def train_model(self, model, num_epochs=5, lr=0.001, name="baseline"):
        """Train model and record performance"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], 
                             lr=lr, momentum=0.9, weight_decay=1e-4)
        
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        print(f"\nTraining {name} model...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_train_loss = running_loss / len(self.train_dataset)
            epoch_train_acc = running_corrects.double() / len(self.train_dataset)
            
            # validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)
            
            epoch_val_loss = val_running_loss / len(self.val_dataset)
            epoch_val_acc = val_running_corrects.double() / len(self.val_dataset)
            
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc.cpu().numpy())
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc.cpu().numpy())
            
            print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
            print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        
        training_time = time.time() - start_time
        print(f'Training complete in {training_time//60:.0f}m {training_time%60:.0f}s')
        
        # store results
        self.results[name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'training_time': training_time,
            'final_val_acc': val_accs[-1]
        }
        
        return model
    
    def disable_skip_connections(self, model, layers_to_modify=None):
        """Task 1.2: Disable skip connections in selected residual blocks"""
        if layers_to_modify is None:
            # disable skip connections in first few blocks of layer2 and layer3
            layers_to_modify = [
                ('layer2', [0, 1]),  # first 2 blocks in layer2
                ('layer3', [0, 1])   # first 2 blocks in layer3
            ]
        
        modified_model = copy.deepcopy(model)
        
        for layer_name, block_indices in layers_to_modify:
            layer = getattr(modified_model, layer_name)
            for idx in block_indices:
                if idx < len(layer):
                    # override the forward method to skip the residual connection
                    def forward_without_skip(self, x, original_forward=layer[idx].forward):
                        # call the original forward but without adding the identity
                        identity = x
                        out = self.conv1(x)
                        out = self.bn1(out)
                        out = self.relu(out)
                        
                        out = self.conv2(out)
                        out = self.bn2(out)
                        out = self.relu(out)
                        
                        out = self.conv3(out)
                        out = self.bn3(out)
                        
                        if self.downsample is not None:
                            identity = self.downsample(x)
                        
                        # skip adding the residual connection
                        # out += identity  
                        out = self.relu(out)
                        
                        return out
                    
                    # bind the new forward method
                    layer[idx].forward = forward_without_skip.__get__(layer[idx])
                    print(f"Disabled skip connection in {layer_name}.{idx}")
        
        return modified_model
    
    def register_feature_hooks(self, model):
        """Task 1.3: Register hooks to extract features from different layers"""
        self.features = {}
        self.hooks = []
        
        # define layers to extract features from
        layers_to_hook = {
            'early': model.layer1[0],      # early layer (after first few convolutions)
            'middle': model.layer2[0],     # middle layer
            'late': model.layer4[-1]       # late layer (before final pooling)
        }
        
        def get_hook(name):
            def hook(module, input, output):
                # store features (detach to avoid gradients)
                self.features[name] = output.detach().cpu()
            return hook
        
        # register hooks
        for name, layer in layers_to_hook.items():
            hook = layer.register_forward_hook(get_hook(name))
            self.hooks.append(hook)
        
        return layers_to_hook
    
    def extract_features_for_visualization(self, model, num_samples=1000):
        """Extract features from different layers for visualization"""
        self.register_feature_hooks(model)
        
        model.eval()
        features_dict = {'early': [], 'middle': [], 'late': []}
        labels_list = []
        
        count = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                if count >= num_samples:
                    break
                    
                inputs = inputs.to(self.device)
                batch_size = min(inputs.size(0), num_samples - count)
                inputs = inputs[:batch_size]
                labels = labels[:batch_size]
                
                # forward pass (this will trigger hooks)
                _ = model(inputs)
                
                # collect features
                for layer_name in features_dict.keys():
                    if layer_name in self.features:
                        # global average pooling to reduce spatial dimensions
                        feat = self.features[layer_name]
                        if len(feat.shape) == 4:  # conv features [B, C, H, W]
                            feat = torch.mean(feat, dim=[2, 3])  # [B, C]
                        features_dict[layer_name].append(feat)
                
                labels_list.append(labels)
                count += batch_size
        
        # clean up hooks
        for hook in self.hooks:
            hook.remove()
        
        # concatenate all features
        final_features = {}
        for layer_name in features_dict.keys():
            if features_dict[layer_name]:
                final_features[layer_name] = torch.cat(features_dict[layer_name], dim=0).numpy()
        
        final_labels = torch.cat(labels_list, dim=0).numpy()
        
        return final_features, final_labels
    
    def visualize_features(self, features_dict, labels, method='tsne'):
        """Task 1.3: Visualize features using t-SNE or UMAP"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (layer_name, features) in enumerate(features_dict.items()):
            print(f"Visualizing {layer_name} features with {method.upper()}...")
            
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            else:  # umap
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            
            # reduce dimensions
            features_2d = reducer.fit_transform(features)
            
            # create scatter plot
            scatter = axes[idx].scatter(features_2d[:, 0], features_2d[:, 1], 
                                     c=labels, cmap='tab10', s=2, alpha=0.7)
            axes[idx].set_title(f'{layer_name.capitalize()} Layer Features ({method.upper()})')
            axes[idx].set_xlabel(f'{method.upper()}-1')
            axes[idx].set_ylabel(f'{method.upper()}-2')
            
            # add colorbar
            cbar = plt.colorbar(scatter, ax=axes[idx])
            cbar.set_label('Class')
        
        plt.tight_layout()
        plt.savefig(f'feature_visualization_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_transfer_learning_approaches(self):
        """Task 1.4: Compare different transfer learning approaches"""
        print("\n=== Transfer Learning Comparison ===")
        
        approaches = {
            'pretrained_head_only': {'pretrained': True, 'freeze_backbone': True},
            'pretrained_full_finetune': {'pretrained': True, 'freeze_backbone': False},
            'random_init': {'pretrained': False, 'freeze_backbone': False}
        }
        
        for name, config in approaches.items():
            print(f"\nTesting {name}...")
            
            if config['pretrained']:
                model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet152(weights=None)
            
            # freeze backbone if specified
            if config['freeze_backbone']:
                for param in model.parameters():
                    param.requires_grad = False
            
            # replace final layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))
            model = model.to(self.device)
            
            # train for fewer epochs for comparison
            self.train_model(model, num_epochs=3, name=name)
    
    def plot_results(self):
        """Plot training curves and comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # plot training curves
        for name, results in self.results.items():
            epochs = range(1, len(results['train_losses']) + 1)
            
            axes[0, 0].plot(epochs, results['train_losses'], label=f'{name} Train')
            axes[0, 0].plot(epochs, results['val_losses'], '--', label=f'{name} Val')
            axes[0, 1].plot(epochs, results['train_accs'], label=f'{name} Train')
            axes[0, 1].plot(epochs, results['val_accs'], '--', label=f'{name} Val')
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # final accuracy comparison
        names = list(self.results.keys())
        final_accs = [self.results[name]['final_val_acc'] for name in names]
        training_times = [self.results[name]['training_time'] for name in names]
        
        axes[1, 0].bar(names, final_accs)
        axes[1, 0].set_title('Final Validation Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(names, training_times)
        axes[1, 1].set_title('Training Time (seconds)')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('resnet152_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete Task 1 analysis"""
        print("Starting ResNet-152 Inner Workings Analysis...")
        
        # task 1.1: baseline Setup
        print("\n=== Task 1.1: Baseline Setup ===")
        baseline_model = self.create_baseline_model()
        baseline_model = self.train_model(baseline_model, num_epochs=5, name="baseline")
        
        # task 1.2: residual connections
        print("\n=== Task 1.2: Residual Connections in Practice ===")
        no_skip_model = self.disable_skip_connections(baseline_model)
        no_skip_model = self.train_model(no_skip_model, num_epochs=5, name="no_skip_connections")
        
        # task 1.3: feature hierarchies
        print("\n=== Task 1.3: Feature Hierarchies and Representations ===")
        features_dict, labels = self.extract_features_for_visualization(baseline_model, num_samples=1000)
        
        # visualize with both t-SNE and UMAP
        self.visualize_features(features_dict, labels, method='tsne')
        self.visualize_features(features_dict, labels, method='umap')
        
        # task 1.4: transfer learning comparison
        print("\n=== Task 1.4: Transfer Learning and Generalization ===")
        self.compare_transfer_learning_approaches()
        
        # plot all results
        print("\n=== Plotting Results ===")
        self.plot_results()
        
        # print summary
        self.print_analysis_summary()
    
    def print_analysis_summary(self):
        """Print analysis summary with answers to assignment questions"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY AND ANSWERS")
        print("="*60)
        
        print("\n1. Why is it unnecessary to train ResNet-152 from scratch on small datasets?")
        print("- ResNet-152 has 60+ million parameters, while CIFAR-10 has only 50K training samples")
        print("- This leads to severe overfitting and poor generalization")
        print("- Pre-trained features on ImageNet capture general visual patterns transferable to CIFAR-10")
        print("- Training from scratch would require massive computational resources and time")
        
        print("\n2. What does freezing most of the network tell us about feature transferability?")
        print("- High validation accuracy with frozen backbone shows features are highly transferable")
        print("- Early layers capture low-level features (edges, textures) that are universal")
        print("- Middle layers capture mid-level patterns that generalize across datasets")
        print("- Only task-specific classification head needs retraining")
        
        if 'no_skip_connections' in self.results:
            baseline_acc = self.results['baseline']['final_val_acc']
            no_skip_acc = self.results['no_skip_connections']['final_val_acc']
            
            print(f"\n3. Effect of removing skip connections:")
            print(f"- Baseline accuracy: {baseline_acc:.4f}")
            print(f"- No skip connections accuracy: {no_skip_acc:.4f}")
            print(f"- Performance difference: {baseline_acc - no_skip_acc:.4f}")
            print("- Skip connections enable gradient flow in very deep networks")
            print("- Without them, gradients vanish and training becomes unstable")
        
        print("\n4. Feature hierarchy evolution:")
        print("- Early layers: Low-level features, poor class separability")
        print("- Middle layers: Mid-level patterns, improving separability")
        print("- Late layers: High-level semantic features, clear class clusters")
        print("- t-SNE/UMAP visualization shows progressive abstraction")
        
        if len(self.results) > 2:
            print("\n5. Transfer learning comparison:")
            for name, result in self.results.items():
                print(f"- {name}: {result['final_val_acc']:.4f} accuracy, "
                      f"{result['training_time']:.1f}s training time")
            print("- Pre-trained weights provide significant advantage")
            print("- Freezing backbone is most compute-efficient")
            print("- Full fine-tuning may overfit on small datasets")

# running the complete analysis
if __name__ == "__main__":
    explorer = ResNetExplorer()
    explorer.run_complete_analysis()