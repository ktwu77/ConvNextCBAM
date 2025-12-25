import os
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import timm
from transformers import set_seed
# ----------------------------- CBAM MODULES -----------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.shared(self.avg_pool(x))
        max_out = self.shared(self.max_pool(x))
        out = torch.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(x_cat))
        return x * out

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ----------------------------- MODEL -----------------------------
class ConvNeXtCBAMClassifier(nn.Module):
    def __init__(self, model_name='convnext_tiny', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        self.cbam = CBAM(self.backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.head(x)
        return x

# ----------------------------- LDAM Loss -----------------------------
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.from_numpy(m_list).float()
        self.s = s
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(self.m_list[None, :].to(x.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - index_float * batch_m
        return self.cross_entropy(self.s * x_m, target).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ----------------------------- DATASET -----------------------------
class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.filenames = []
        self.transform = transform
        label_map = {'retear': 0, 'heal': 1}

        for label_name in ['retear', 'heal']:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_map[label_name])
                    self.filenames.append(fname)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return {'input': image, 'label': torch.tensor(label, dtype=torch.long), 'filename': self.filenames[idx]}


# ----------------------------- HELPERS -----------------------------
def compute_class_weights(labels):
    class_sample_counts = np.bincount(labels)
    weight_per_class = 1. / class_sample_counts
    return weight_per_class[labels], class_sample_counts

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    preds, trues, probs, fnames, total_loss = [], [], [], [], 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            filenames = batch['filename']
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            probs_batch = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds_batch = torch.argmax(logits, dim=1).cpu().numpy()
            trues_batch = labels.cpu().numpy()

            preds.extend(preds_batch)
            trues.extend(trues_batch)
            probs.extend(probs_batch)
            fnames.extend(filenames)

    return np.array(preds), np.array(trues), np.array(probs), fnames, total_loss / len(dataloader)

# ----------------------------- MAIN ENTRY -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--epochs-stage1', type=int, default=5)
    parser.add_argument('--epochs-stage2', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=15, shear=10),  # rotation ±15°, shear ±10°
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = FolderDataset(os.path.join(args.root_dir, 'train'), transform=train_transform)
    val_set = FolderDataset(os.path.join(args.root_dir, 'valid'), transform=val_transform)
    test_set = FolderDataset(os.path.join(args.root_dir, 'test'), transform=val_transform)

    train_weights, cls_num_list = compute_class_weights(train_set.labels)
    val_weights, _ = compute_class_weights(val_set.labels)

    train_loader = DataLoader(train_set, sampler=WeightedRandomSampler(train_weights, len(train_weights)), batch_size=64)
    val_loader = DataLoader(val_set, sampler=WeightedRandomSampler(val_weights, len(val_weights)), batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    seed_list = [3, 7, 21, 42]
    models = []
    results = []

    for seed in seed_list:
        set_seed(seed)
        model = ConvNeXtCBAMClassifier().to(device)

        # Stage 1 - LDAM Loss
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        for p in model.backbone.stages[-1].parameters(): p.requires_grad = True
        for p in model.cbam.parameters(): p.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        ldam_loss_fn = LDAMLoss(cls_num_list)

        print(f"\n[Seed {seed}] Stage 1 - LDAM Loss")
        for epoch in range(args.epochs_stage1):
            model.train()
            running_loss = 0
            for batch in tqdm(train_loader, desc=f'Stage 1 Epoch {epoch + 1}/{args.epochs_stage1}'):
                optimizer.zero_grad()
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = ldam_loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"  [Epoch {epoch + 1}] Avg LDAM Loss: {avg_loss:.4f}")

        # Stage 2 - CrossEntropy Loss
        for p in model.parameters(): p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        focal_loss_fn = FocalLoss(alpha=1, gamma=2)

        print(f"\n[Seed {seed}] Stage 2 - Focal Loss")
        for epoch in range(args.epochs_stage2):
            model.train()
            running_loss = 0
            for batch in tqdm(train_loader, desc=f'Stage 2 Epoch {epoch + 1}/{args.epochs_stage2}'):
                optimizer.zero_grad()
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = focal_loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"  [Epoch {epoch + 1}] Avg Focal Loss: {avg_loss:.4f}")

        preds, trues, probs, fnames, _ = evaluate(model, test_loader, focal_loss_fn, device)
        auc = roc_auc_score(trues, probs)
        f1 = f1_score(trues, preds, average='micro')
        results.append({'Seed': seed, 'F1': f1, 'AUC': auc})

        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_seed{seed}.pth'))

        pd.DataFrame({
            'filename': fnames,
            'true_label': trues,
            'pred_label': preds,
            'prob_heal': probs
        }).to_csv(os.path.join(args.output_dir, f'predictions_seed{seed}.csv'), index=False)

    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, 'results_summary.csv'), index=False)

    # Ensemble evaluation
    ensemble_probs = []
    ensemble_trues = None

    for seed in seed_list:
        df_pred = pd.read_csv(os.path.join(args.output_dir, f'predictions_seed{seed}.csv'))
        if ensemble_trues is None:
            ensemble_trues = df_pred['true_label'].values
        ensemble_probs.append(df_pred['prob_heal'].values)

    ensemble_probs = np.mean(ensemble_probs, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    ensemble_auc = roc_auc_score(ensemble_trues, ensemble_probs)
    ensemble_f1 = f1_score(ensemble_trues, ensemble_preds, average='micro')

    pd.DataFrame({
        'true_label': ensemble_trues,
        'ensemble_prob_heal': ensemble_probs,
        'ensemble_pred_label': ensemble_preds
    }).to_csv(os.path.join(args.output_dir, 'ensemble_predictions.csv'), index=False)

    df = pd.read_csv(os.path.join(args.output_dir, 'results_summary.csv'))
    df = pd.concat([df, pd.DataFrame([{'Seed': 'Ensemble', 'F1': ensemble_f1, 'AUC': ensemble_auc}])], ignore_index=True)
    df.to_csv(os.path.join(args.output_dir, 'results_summary.csv'), index=False)

    # Confusion matrix and ROC
    cm = confusion_matrix(ensemble_trues, ensemble_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retear', 'Heal'])
    disp.plot(cmap='Blues')
    plt.title(f'Ensemble Confusion Matrix\nF1={ensemble_f1:.2f}, AUC={ensemble_auc:.2f}')
    plt.savefig(os.path.join(args.output_dir, 'ensemble_confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(ensemble_trues, ensemble_probs)
    plt.plot(fpr, tpr, label=f'Ensemble AUC={ensemble_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title('Ensemble ROC Curve')
    plt.savefig(os.path.join(args.output_dir, 'ensemble_roc_curve.png'))
    plt.close()

    # Save averaged ensemble model
    avg_state_dict = {}
    state_dicts = [torch.load(os.path.join(args.output_dir, f'model_seed{seed}.pth'), map_location=device) for seed in seed_list]
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)

    final_model = ConvNeXtCBAMClassifier().to(device)
    final_model.load_state_dict(avg_state_dict)
    torch.save(final_model.state_dict(), os.path.join(args.output_dir, 'ensemble_model.pth'))
