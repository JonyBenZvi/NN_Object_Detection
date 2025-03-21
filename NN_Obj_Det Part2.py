import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader #, random_split
from torchvision import models, transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ==========================
# 1. Model Architecture
# ==========================

# ResNet18 Backbone Feature Extractor
class ResNet18Backbone(nn.Module):
    def __init__(self):
        super(ResNet18Backbone, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove fully connected layer: keep convolutional feature extractor
        self.features = nn.Sequential(
            resnet.conv1,  
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  # Output: (batch, 512, H, W) with H,W ~7 for 224x224 input
        )
        
    def forward(self, x):
        return self.features(x)

# Detection Head to predict [center_x, center_y, width, height, confidence]
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_outputs=5):
        super(DetectionHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_outputs, kernel_size=1)
        )
        # Use Adaptive Average Pooling to collapse spatial dimensions for a single prediction per image
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.conv(x)      # shape: (batch, num_outputs, H, W)
        x = self.pool(x)      # shape: (batch, num_outputs, 1, 1)
        x = x.view(x.size(0), -1)  # shape: (batch, num_outputs)
        return x

# Combined Object Detection Model
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = ResNet18Backbone()
        self.head = DetectionHead(in_channels=512, num_outputs=5)
        
    def forward(self, x):
        features = self.backbone(x)
        detection = self.head(features)
        return detection

# ==========================
# 2. Dataset and Data Augmentation
# ==========================
class VehiclesDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        """
        Args:
            images_dir (str): Directory with images.
            annotations_dir (str): Directory with annotation text files.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.annotations_paths = sorted(glob.glob(os.path.join(annotations_dir, "*.txt")))
        self.transform = transform
        
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load annotation from corresponding txt file
        base = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(os.path.dirname(self.annotations_paths[0]), base + ".txt")
        with open(ann_path, "r") as f:
            line = f.readline().strip()
            # Expecting: center_x center_y width height confidence
            bbox = np.array([float(x) for x in line.split()], dtype=np.float32)
        
        # Apply data augmentation / transformation
        if self.transform:
            image = self.transform(image)
        
        # Return image and bounding box (as tensor)
        target = torch.tensor(bbox, dtype=torch.float32)
        return image, target

# Define data augmentation and preprocessing transforms
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# 3. Loss Functions and Metrics
# ==========================

def detection_loss(predictions, targets):
    """
    Combines MSE loss for bounding box regression and BCE loss for confidence.
    predictions: [center_x, center_y, width, height, confidence]
    targets: same format
    """
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Regression loss for bbox coordinates
    reg_loss = mse_loss(predictions[:, :4], targets[:, :4])
    
    # Confidence loss (target confidence is assumed to be 1)
    conf_pred = torch.sigmoid(predictions[:, 4])
    conf_target = targets[:, 4]
    conf_loss = bce_loss(conf_pred, conf_target)
    
    return reg_loss + conf_loss

def iou_metric(pred_box, true_box):
    """
    Calculate Intersection over Union (IoU) for a single prediction and true box.
    Boxes are expected in format [center_x, center_y, width, height] in normalized coordinates.
    """
    # Convert from center to (x1, y1, x2, y2)
    def to_corners(box):
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2
    
    p = to_corners(pred_box)
    t = to_corners(true_box)
    
    # Intersection rectangle
    inter_x1 = max(p[0], t[0])
    inter_y1 = max(p[1], t[1])
    inter_x2 = min(p[2], t[2])
    inter_y2 = min(p[3], t[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Areas of prediction and true boxes
    area_p = (p[2] - p[0]) * (p[3] - p[1])
    area_t = (t[2] - t[0]) * (t[3] - t[1])
    
    union_area = area_p + area_t - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

# ==========================
# 4. Data Partitioning and DataLoaders
# ==========================
def create_dataloaders(root_dir, batch_size=8, val_split=0.2):
    # Directories for training and validation
    train_images_dir = os.path.join(root_dir, "train", "images")
    train_ann_dir = os.path.join(root_dir, "train", "annotations")
    val_images_dir = os.path.join(root_dir, "val", "images")
    val_ann_dir = os.path.join(root_dir, "val", "annotations")
    
    train_dataset = VehiclesDataset(train_images_dir, train_ann_dir, transform=train_transforms)
    val_dataset = VehiclesDataset(val_images_dir, val_ann_dir, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# ==========================
# 5. Training Loop with TensorBoard Monitoring
# ==========================
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    writer = SummaryWriter("runs/vehicle_detection")
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = detection_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        iou_scores = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = detection_loss(outputs, targets)
                val_loss += loss.item()
                
                # Calculate IoU for each sample
                for pred, true in zip(outputs, targets):
                    # Convert predictions and true boxes (first 4 values)
                    pred_box = pred[:4].cpu().numpy()
                    true_box = true[:4].cpu().numpy()
                    iou = iou_metric(pred_box, true_box)
                    iou_scores.append(iou)
                    
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = np.mean(iou_scores)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/IoU", avg_iou, epoch)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Avg IoU: {avg_iou:.4f}")
    
    writer.close()

# ==========================
# 6. Main Execution
# ==========================
if __name__ == "__main__":
    # Root directory where the dataset is located
    dataset_root = "/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/dataset"  # TODO: Update
    train_loader, val_loader = create_dataloaders(dataset_root, batch_size=8)
    
    model = ObjectDetectionModel()
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4)
    
    # Save the trained model
    torch.save(model.state_dict(), "object_detection_model.pth")
