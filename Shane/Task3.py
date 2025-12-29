import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


# ========================
# Dataset preparation
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels, img_name # Returning img_name for submission mapping


# ========================
# build model
# ========================
def build_model(backbone="resnet18", num_classes=3, pretrained=True, attention_type="se"):

    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # Add attention to final layer
        if attention_type == "se":
            model.layer4[1].se = SqueezeExcitation(512)
        elif attention_type == "mha":
            model.layer4[1].mha = MultiHeadAttention(512, num_heads=4)
        
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if attention_type == "se":
            model.features[8].se = SqueezeExcitation(320)
        elif attention_type == "mha":
            model.features[8].mha = MultiHeadAttention(320, num_heads=4)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model

# ========================
# Task 2.1: Focal Loss Implementation
# ========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Focal Loss for addressing class imbalance
        
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels (0 or 1)
        
        Returns:
            Focal loss value
        """
        # Get sigmoid probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal loss
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()
    
# ========================
# Task 2.2: Class-Balanced Loss Implementation
# ========================
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class=None, beta=0.9999):
        """
        Class-Balanced Loss for addressing class imbalance
        
        Args:
            samples_per_class: List/array of number of samples per class. 
                              If None, will be calculated from data
            beta: Re-weighting parameter in range (0,1). Higher values give more weight to rare classes.
                  Typical value: 0.9999
        """
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.weights = None
        
        if samples_per_class is not None:
            self._calculate_weights()

    def _calculate_weights(self):
        """Calculate class weights based on effective number of samples"""
        if self.samples_per_class is None:
            return
        
        effective_num = 1.0 - torch.pow(self.beta, torch.tensor(self.samples_per_class, dtype=torch.float32))
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize
     
        # CHANGED: Use try-except to handle re-registration
        try:
            self.register_buffer('weights', weights)
        except KeyError:
            # If buffer already exists, update it directly
            self.weights = weights

    def forward(self, inputs, targets, samples_per_class=None):
        """
        Args:
            inputs: Model predictions (logits), shape: (batch_size, num_classes)
            targets: Ground truth labels (0 or 1), shape: (batch_size, num_classes)
            samples_per_class: Optional, to update weights dynamically
        
        Returns:
            Class-balanced loss value
        """
        # Update weights if provided
        if samples_per_class is not None and self.weights is None:
            self.samples_per_class = samples_per_class
            self._calculate_weights()
        
        # Standard binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply class weights
        if self.weights is not None:
            # For each sample, compute weighted loss considering all classes
            weighted_loss = bce_loss * self.weights.to(inputs.device).unsqueeze(0)
            loss = weighted_loss.mean()
        else:
            loss = bce_loss.mean()
        
        return loss
    
# ========================
# Task 2.2: Hybrid Loss (Focal + Class-Balanced)
# ========================
class HybridLoss(nn.Module):
    def __init__(self, samples_per_class=None, alpha=0.25, gamma=2.0, beta=0.9999, lambda_weight=0.5):
        """
        Hybrid Loss combining Focal Loss and Class-Balanced Loss
        
        Args:
            samples_per_class: List of samples per class for class-balanced weighting
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            beta: Class-balanced loss beta parameter
            lambda_weight: Weight to balance between focal (λ) and class-balanced (1-λ) losses
                          Range [0, 1]: 0.5 means equal weight to both losses
        """
        super(HybridLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.class_balanced_loss = ClassBalancedLoss(samples_per_class=samples_per_class, beta=beta)
        self.lambda_weight = lambda_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels (0 or 1)
        
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        class_balanced = self.class_balanced_loss(inputs, targets)
        
        # Weighted combination
        hybrid_loss = self.lambda_weight * focal + (1 - self.lambda_weight) * class_balanced
        
        return hybrid_loss
    
# =========================
# Task 3.1: Squeeze-and-Excitation (SE) Block
# ========================
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        excitation = self.fc(squeeze).view(b, c, 1, 1)
        return x * excitation

# ========================
# Task 3.2: Multi Head Attention (MHA) Block
# ========================
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Block"""
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Generate queries, keys and values
        queries = self.query_conv(x).view(b, self.num_heads, self.head_dim, h * w)
        keys = self.key_conv(x).view(b, self.num_heads, self.head_dim, h * w)
        values = self.value_conv(x).view(b, self.num_heads, self.head_dim, h * w)

        # Scaled dot-product attention
        scores = torch.matmul(queries.transpose(-2, -1), keys) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, values.transpose(-2, -1)).transpose(-2, -1).contiguous()
        out = out.view(b, c, h, w)

        out = self.out_conv(out)
        return out
    
# ========================
# model training and val
# ========================
def train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, onsite_image_dir,
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints",pretrained_backbone=None,
                       loss_type="focal", attention_type="se"):
    """
    Args:
        loss_type: "focal", "class_balanced", or "hybrid" to choose loss function
        attention_type: "se" or "mha" to choose attention mechanism
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)
    onsite_ds = RetinaMultiLabelDataset(onsite_csv, onsite_image_dir, transform)  # Using test set as onsite for demo

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False, attention_type=attention_type).to("cuda" if torch.cuda.is_available() else "cpu")

    for p in model.parameters():
        p.requires_grad = True
    
    # loss & optimizer
     # loss & optimizer - CHANGED: Support all three loss types
    if loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_type == "class_balanced":
        train_data = pd.read_csv(train_csv)
        samples_per_class = train_data.iloc[:, 1:].sum().values
        criterion = ClassBalancedLoss(samples_per_class=samples_per_class.tolist(), beta=0.9999)
    elif loss_type == "hybrid":
        train_data = pd.read_csv(train_csv)
        samples_per_class = train_data.iloc[:, 1:].sum().values
        criterion = HybridLoss(samples_per_class=samples_per_class.tolist(), 
                              alpha=0.25, gamma=2.0, beta=0.9999, lambda_weight=0.5)
    else:
        raise ValueError("loss_type must be 'focal', 'class_balanced', or 'hybrid'")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state_dict, strict=False)
    

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")

    # ========================
    # testing
    # ========================
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = torch.tensor(y_true).numpy()
    y_pred = torch.tensor(y_pred).numpy()

    disease_names = ["DR", "Glaucoma", "AMD"]

    for i, disease in enumerate(disease_names):  #compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro",zero_division=0)
        recall = recall_score(y_t, y_p, average="macro",zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro",zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")
    
    # ========================
    # Inference on Onsite Test (For Submission)
    # ========================
    print("\n>>> Generating Submission for Onsite Test Set...")
    onsite_results = []

    with torch.no_grad():
        for imgs, _, img_names in tqdm(onsite_loader, desc="Onsite Inference"):
            imgs = imgs.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            # Binarize predictions for submission (0 or 1)
            preds = (probs > 0.5).astype(int)

            for i in range(len(img_names)):
                # id, D, G, A
                row = [img_names[i], preds[i][0], preds[i][1], preds[i][2]]
                onsite_results.append(row)

    # Save submission file
    submission_df = pd.DataFrame(onsite_results, columns=['id', 'D', 'G', 'A'])
    submission_df.to_csv("submission.csv", index=False)
    print(">>> 'submission.csv' saved successfully!")

    
# ========================
# main
# ========================
if __name__ == "__main__":
    train_csv = "train.csv" # replace with your own train label file path
    val_csv   = "val.csv" # replace with your own validation label file path
    test_csv  = "offsite_test.csv"  # replace with your own test label file path
    onsite_csv = "onsite_test_submission.csv"  # replace with your own onsite test label file path
    train_image_dir ="./images/train"   # replace with your own train image floder path
    val_image_dir = "./images/val"  # replace with your own validation image floder path
    test_image_dir = "./images/offsite_test" # replace with your own test image floder path
    onsite_image_dir = "./images/onsite_test" # replace with your own onsite test image floder path
    pretrained_backbone = './pretrained_backbone/ckpt_resnet18_ep50.pt'  # replace with your own pretrained backbone path
    backbone = 'resnet18'  # backbone choices: ["resnet18", "efficientnet"]

    # Test with Squeeze-and-Excitation Attention
    print("\n" + "=" * 50)
    print("Training with Squeeze-and-Excitation Attention...")
    print("=" * 50)
    train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, onsite_image_dir,
                           epochs=20, batch_size=32, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone,
                           loss_type="hybrid", attention_type="se")
    
    # Test with Multi-Head Attention
    print("\n" + "=" * 50)
    print("Training with Multi-Head Attention...")
    print("=" * 50)
    train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, onsite_image_dir,
                           epochs=20, batch_size=32, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone,
                           loss_type="hybrid", attention_type="mha")