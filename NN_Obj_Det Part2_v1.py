print("Loading imports")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================
# Constans & Initializations
# ==========================
EPSILON = 1e-6
ADAM_STEP = 5e-5 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 416
# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

# Batch size for training 
batch_size = 32

# Number of epochs for training 
epochs = 20

# Grid cell sizes 
s = [IMG_SIZE // 32, IMG_SIZE // 16, IMG_SIZE // 8] 

# Number of classes 
n_classes = 20

# Class labels
class_labels = [ 
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# ==========================
#         Auxilary
# ==========================

# Non-maximum suppression function to remove overlapping bounding boxes 
def nms(bboxes, iou_threshold, threshold): 
    # Filter out bounding boxes with confidence below the threshold. 
    bboxes = [box for box in bboxes if box[1] > threshold] 
  
    # Sort the bounding boxes by confidence in descending order. 
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 
  
    # Initialize the list of bounding boxes after non-maximum suppression. 
    bboxes_nms = [] 
  
    while bboxes: 
        # Get the first bounding box. 
        first_box = bboxes.pop(0) 
  
        # Iterate over the remaining bounding boxes. 
        for box in bboxes: 
        # If the bounding boxes do not overlap or if the first bounding box has 
        # a higher confidence, then add the second bounding box to the list of 
        # bounding boxes after non-maximum suppression. 
            if box[0] != first_box[0] or iou(
                torch.tensor(first_box[2:]), 
                torch.tensor(box[2:]) ) < iou_threshold: 
                # Check if box is not in bboxes_nms 
                if box not in bboxes_nms: 
                    # Add box to bboxes_nms 
                    bboxes_nms.append(box) 
  
    # Return bounding boxes after non-maximum suppression. 
    return bboxes_nms


# Function to convert cells to bounding boxes 
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
	# Batch size used on predictions 
	batch_size = predictions.shape[0] 
	# Number of anchors 
	num_anchors = len(anchors) 
	# List of all the predictions 
	box_predictions = predictions[..., 1:5] 

	# If the input is predictions then we will pass the x and y coordinate 
	# through sigmoid function and width and height to exponent function and 
	# calculate the score and best class. 
	if is_predictions: 
		anchors = anchors.reshape(1, len(anchors), 1, 1, 2) 
		box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) 
		box_predictions[..., 2:] = torch.exp( 
			box_predictions[..., 2:]) * anchors 
		scores = torch.sigmoid(predictions[..., 0:1]) 
		best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 
	
	# Else we will just calculate scores and best class. 
	else: 
		scores = predictions[..., 0:1] 
		best_class = predictions[..., 5:6] 

	# Calculate cell indices 
	cell_indices = ( 
		torch.arange(s) 
		.repeat(predictions.shape[0], 3, s, 1) 
		.unsqueeze(-1) 
		.to(predictions.device) 
	) 

	# Calculate x, y, width and height with proper scaling 
	x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
	y = 1 / s * (box_predictions[..., 1:2] +
				cell_indices.permute(0, 1, 3, 2, 4)) 
	width_height = 1 / s * box_predictions[..., 2:4] 

	# Concatinating the values and reshaping them in 
	# (BATCH_SIZE, num_anchors * S * S, 6) shape 
	converted_bboxes = torch.cat( 
		(best_class, scores, x, y, width_height), dim=-1
	).reshape(batch_size, num_anchors * s * s, 6) 

	# Returning the reshaped and converted bounding box list 
	return converted_bboxes.tolist()


# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes): 
	# Getting the color map from matplotlib 
	colour_map = plt.get_cmap("tab20b") 
	# Getting 21 different colors from the color map for 21 different classes 
	colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 

	# Reading the image with OpenCV 
	img = np.array(image) 
	# Getting the height and width of the image 
	h, w, _ = img.shape 

	# Create figure and axes 
	fig, ax = plt.subplots(1) 

	# Add image to plot 
	ax.imshow(img) 

	# Plotting the bounding boxes and labels over the image 
	for box in boxes: 
		# Get the class from the box 
		class_pred = box[0] 
		# Get the center x and y coordinates 
		box = box[2:] 
		# Get the upper left corner coordinates 
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2

		# Create a Rectangle patch with the bounding box 
		rect = patches.Rectangle( 
			(upper_left_x * w, upper_left_y * h), 
			box[2] * w, 
			box[3] * h, 
			linewidth=2, 
			edgecolor=colors[int(class_pred)], 
			facecolor="none", 
		) 
		
		# Add the patch to the Axes 
		ax.add_patch(rect) 
		
		# Add class name to the patch 
		plt.text( 
			upper_left_x * w, 
			upper_left_y * h, 
			s=class_labels[int(class_pred)], 
			color="white", 
			verticalalignment="top", 
			bbox={"color": colors[int(class_pred)], "pad": 0}, 
		) 

	# Display the plot 
	plt.show()


# Function to save checkpoint 
def save_checkpoint(model, optimizer, filename="best_model_checkpoint.pth.tar"): 
    print("==> Saving checkpoint") 
    checkpoint = { 
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
    } 
    torch.save(checkpoint, filename)



# Function to load checkpoint 
def load_checkpoint(checkpoint_file, model, optimizer, lr): 
    print("==> Loading checkpoint") 
    checkpoint = torch.load(checkpoint_file, map_location=device) 
    model.load_state_dict(checkpoint["state_dict"]) 
    optimizer.load_state_dict(checkpoint["optimizer"]) 
  
    for param_group in optimizer.param_groups: 
        param_group["lr"] = lr 

# ==========================
# 1. Model Architecture
# ==========================

# ResNet18 Backbone with Multi-Scale Outputs
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else 'DEFAULT')
        # Use initial layers (layer0): conv, bn, relu, and maxpool.
        self.layer0 = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool
        )
        self.layer1 = resnet.layer1   # Output: (batch, 64, H/4, W/4)
        self.layer2 = resnet.layer2   # Output: (batch, 128, H/8, W/8)  → small scale
        self.layer3 = resnet.layer3   # Output: (batch, 256, H/16, W/16) → medium scale
        self.layer4 = resnet.layer4   # Output: (batch, 512, H/32, W/32) → large scale

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        feat_small = self.layer2(x)   # lowest-level features (higher resolution)
        feat_medium = self.layer3(feat_small)
        feat_large = self.layer4(feat_medium)
        return feat_small, feat_medium, feat_large


# Detection Head to predict [center_x, center_y, width, height, confidence]
class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        """
        Args:
            in_channels (int): number of input channels from the backbone feature map.
            num_anchors (int): number of anchors used for this scale.
            num_classes (int): number of object classes.
        """
        super(YOLODetectionHead, self).__init__()
        # YOLO outputs for each anchor: 5 + num_classes values
        out_channels = num_anchors * (num_classes + 5)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.num_classes = num_classes

    # Defining the forward pass and reshaping the output to the desired output  
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
    def forward(self, x):
        output = self.conv(x) 
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output = output.permute(0, 1, 3, 4, 2) 
        return output

# Combined Object Detection Model
class ObjectDetectionModel(nn.Module):
    def __init__(self, num_anchors=3, num_classes=n_classes):
        """
        Args:
            num_anchors (int): number of anchors per scale.
            num_classes (int): number of object classes.
        """
        super(ObjectDetectionModel, self).__init__()
        self.backbone = ResNet18Backbone(pretrained=True)
        # Create detection heads for three scales.
        # Here we assume:
        #   - large scale features: from layer2 (128 channels)
        #   - medium scale features: from layer3 (256 channels)
        #   - small scale features: from layer4 (512 channels)
        self.head_large = YOLODetectionHead(128, num_anchors, num_classes)
        self.head_medium = YOLODetectionHead(256, num_anchors, num_classes)
        self.head_small = YOLODetectionHead(512, num_anchors, num_classes)

    def forward(self, x):
        feat_large, feat_medium, feat_small = self.backbone(x)
        out_large = self.head_large(feat_large)
        out_medium = self.head_medium(feat_medium)
        out_small = self.head_small(feat_small)
        return out_small, out_medium, out_large

# ==========================
# 2. Dataset and Data Augmentation
# ==========================
class Dataset(Dataset):
    def __init__( 
        self, image_dir, label_dir, anchors,  
        image_size=IMG_SIZE, grid_sizes=s, 
        num_classes=n_classes, transform=None
    ): 
        # Image and label directories 
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        # Image size 
        self.image_size = image_size 
        # Transformations 
        self.transform = transform 
        # Grid sizes for each scale 
        self.grid_sizes = grid_sizes 
        # Anchor boxes 
        self.anchors = torch.tensor( 
            anchors[0] + anchors[1] + anchors[2]) 
        # Number of anchor boxes  
        self.num_anchors = self.anchors.shape[0] 
        # Number of anchor boxes per scale 
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes 
        self.num_classes = num_classes 
        # Ignore IoU threshold 
        self.ignore_iou_thresh = 0.5
  
    def __len__(self): 
        return len(self.label_list) 
      
    def __getitem__(self, idx): 
        # Getting the label path 
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) 
        # We are applying roll to move class label to the last column 
        # 5 columns: x, y, width, height, class_label 
        bboxes = np.roll(np.loadtxt(fname=label_path, 
                         delimiter=" ", ndmin=2), 4, axis=1).tolist() 
          
        # Getting the image path 
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
        image = np.array(Image.open(img_path).convert("RGB")) 
  
        # Albumentations augmentations 
        if self.transform: 
            augs = self.transform(image=image, bboxes=bboxes) 
            image = augs["image"] 
            bboxes = augs["bboxes"] 
  
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
        # target : [probabilities, x, y, width, height, class_label] 
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
                   for s in self.grid_sizes] 
          
        # Identify anchor box and cell for each bounding box 
        for box in bboxes: 
            # Calculate iou of bounding box with anchor boxes 
            iou_anchors = iou(torch.tensor(box[2:4]),  
                              self.anchors,  
                              is_pred=False) 
            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
            x, y, width, height, class_label = box 
  
            # At each scale, assigning the bounding box to the  
            # best matching anchor box 
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                  
                # Identifying the grid size for the scale 
                s = self.grid_sizes[scale_idx] 
                  
                # Identifying the cell to which the bounding box belongs 
                i, j = int(s * y), int(s * x) 
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                  
                # Check if the anchor box is already assigned 
                if not anchor_taken and not has_anchor[scale_idx]: 
  
                    # Set the probability to 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
  
                    # Calculating the center of the bounding box relative 
                    # to the cell 
                    x_cell, y_cell = s * x - j, s * y - i  
  
                    # Calculating the width and height of the bounding box  
                    # relative to the cell 
                    width_cell, height_cell = (width * s, height * s) 
  
                    # Idnetify the box coordinates 
                    box_coordinates = torch.tensor( 
                                        [x_cell, y_cell, width_cell,  
                                         height_cell] 
                                    ) 
  
                    # Assigning the box coordinates to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
  
                    # Assigning the class label to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 
  
                    # Set the anchor box as assigned for the scale 
                    has_anchor[scale_idx] = True
  
                # If the anchor box is already assigned, check if the  
                # IoU is greater than the threshold 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
                    # Set the probability to -1 to ignore the anchor box 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
  
        # Return the image and the target 
        return image, tuple(targets)

# Define data augmentation and preprocessing transforms
train_transforms = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=IMG_SIZE), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Random color jittering 
        A.ColorJitter( 
            brightness=0.5, contrast=0.5, 
            saturation=0.5, hue=0.5, p=0.5
        ), 
        # Flip the image horizontally 
        A.HorizontalFlip(p=0.5), 
        # Normalize the image 
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ],  
    # Augmentation for bounding boxes 
    bbox_params=A.BboxParams( 
                    format="yolo",  
                    min_visibility=0.4,  
                    label_fields=[] 
                ) 
)

val_transforms = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=IMG_SIZE), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Normalize the image 
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ], 
    # Augmentation for bounding boxes  
    bbox_params=A.BboxParams( 
                    format="yolo",  
                    min_visibility=0.4,  
                    label_fields=[] 
                ) 
)

# ==========================
# 3. Loss and Metrics
# ==========================
class Loss(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.mse = nn.MSELoss() 
        self.bce = nn.BCEWithLogitsLoss() 
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.sigmoid = nn.Sigmoid() 
      
    def forward(self, pred, target, anchors): 
        # Identifying which cells in target have objects  
        # and which have no objects 
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0
  
        # Calculating No object loss 
        no_object_loss = self.bce( 
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), 
        ) 
  
          
        # Reshaping anchors to match predictions 
        anchors = anchors.reshape(1, 3, 1, 1, 2) 
        # Box prediction confidence 
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), 
                               torch.exp(pred[..., 3:5]) * anchors 
                            ],dim=-1) 
        # Calculating intersection over union for prediction and target 
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach() 
        # Calculating Object loss 
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), 
                               ious * target[..., 0:1][obj]) 
  
          
        # Predicted box coordinates 
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) 
        # Target box coordinates 
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors) 
        # Calculating box coordinate loss 
        box_loss = self.mse(pred[..., 1:5][obj], 
                            target[..., 1:5][obj]) 
  
          
        # Claculating class loss 
        class_loss = self.cross_entropy((pred[..., 5:][obj]), 
                                   target[..., 5][obj].long()) 
  
        # Total loss 
        return ( 
            box_loss 
            + object_loss 
            + no_object_loss 
            + class_loss 
        )


def iou(box1, box2, is_pred=True):
    """
    Calculate Intersection over Union (IoU) for a single prediction (box1) and true box (box2).
    Boxes are in the format [center_x, center_y, width, height] (normalized coordinates).
    """
    if is_pred: 
        # IoU score for prediction and label 
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format 
          
        # Box coordinates of prediction 
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
  
        # Box coordinates of ground truth 
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
  
        # Get the coordinates of the intersection rectangle 
        x1 = torch.max(b1_x1, b2_x1) 
        y1 = torch.max(b1_y1, b2_y1) 
        x2 = torch.min(b1_x2, b2_x2) 
        y2 = torch.min(b1_y2, b2_y2) 
        # Make sure the intersection is at least 0 
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 
  
        # Calculate the union area 
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
        union = box1_area + box2_area - intersection 
  
        # Calculate the IoU score 
        iou_score = intersection / (union + EPSILON) 
  
        # Return IoU score 
        return iou_score 
      
    else: 
        # IoU score based on width and height of bounding boxes 
          
        # Calculate intersection area 
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 
  
        # Calculate union area 
        box1_area = box1[..., 0] * box1[..., 1] 
        box2_area = box2[..., 0] * box2[..., 1] 
        union_area = box1_area + box2_area - intersection_area 
  
        # Calculate IoU score 
        iou_score = intersection_area / (union_area + EPSILON)
  
        # Return IoU score 
        return iou_score

# ==========================
# 4. Data Partitioning and DataLoaders
# ==========================
#def create_dataloaders(root_dir, batch_size=8):
#    # Directories for training and validation within the dataset
#    train_images_dir = os.path.join(root_dir, "test", "img")
#    train_ann_dir = os.path.join(root_dir, "test", "ann")
#    val_images_dir = os.path.join(root_dir, "val", "img")
#    val_ann_dir = os.path.join(root_dir, "val", "ann")
#       
#    train_dataset = Dataset(train_images_dir, train_ann_dir, anchors=ANCHORS, transform=train_transforms)
#    val_dataset = Dataset(val_images_dir, val_ann_dir, anchors=ANCHORS, transform=val_transforms)
#    
#    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#    
#    return train_loader, val_loader

# ==========================
# 5. Training Loop with TensorBoard Monitoring
# ==========================
#def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
#    model = model.to(device)
#    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#    loss_fn = Loss()
#
#    writer = SummaryWriter("/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/runs/duck_detection")
#    
#    global_step = 0
#    for epoch in range(num_epochs):
#        model.train()
#        running_loss = 0.0
#        
#        for images, targets in train_loader:
#            images = images.to(device)
#            targets = targets.to(device)
#            
#            optimizer.zero_grad()
#            outputs = model(images)
#            loss = loss_fn(outputs, targets)
#            loss.backward()
#            optimizer.step()
#            
#            running_loss += loss.item()
#            writer.add_scalar("Train/Loss", loss.item(), global_step)
#            global_step += 1
#            
#        avg_train_loss = running_loss / len(train_loader)
#        
#        # Evaluate on validation set
#        model.eval()
#        val_loss = 0.0
#        iou_scores = []
#        with torch.no_grad():
#            for images, targets in val_loader:
#                images = images.to(device)
#                targets = targets.to(device)
#                outputs = model(images)
#                loss = loss_fn(outputs, targets)
#                val_loss += loss.item()
#                
#                # Calculate IoU for each sample
#                for pred, true in zip(outputs, targets):
#                    pred_box = pred[:4].cpu().numpy()
#                    true_box = true[:4].cpu().numpy()
#                    iou = iou(pred_box, true_box)
#                    iou_scores.append(iou)
#                    
#        avg_val_loss = val_loss / len(val_loader)
#        avg_iou = np.mean(iou_scores)
#        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
#        writer.add_scalar("Val/IoU", avg_iou, epoch)
#        
#        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Avg IoU: {avg_iou:.4f}")
#    
#    writer.close()


# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True) 
  
    # Initializing a list to store the losses 
    losses = [] 
  
    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device) 
        y0, y1, y2 = ( 
            y[0].to(device), 
            y[1].to(device), 
            y[2].to(device), 
        ) 
  
        with torch.amp.autocast(): 
            # Getting the model predictions 
            outputs = model(x) 
            # Calculating the loss at each scale 
            loss = ( 
                  loss_fn(outputs[0], y0, scaled_anchors[0]) 
                + loss_fn(outputs[1], y1, scaled_anchors[1]) 
                + loss_fn(outputs[2], y2, scaled_anchors[2]) 
            ) 
  
        # Add the loss to the list 
        losses.append(loss.item()) 
  
        # Reset gradients 
        optimizer.zero_grad() 
  
        # Backpropagate the loss 
        scaler.scale(loss).backward() 
  
        # Optimization step 
        scaler.step(optimizer) 
  
        # Update the scaler for next iteration 
        scaler.update() 
  
        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)


def train_model(model, num_epochs=20, learning_rate=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Creating the model from YOLOv3 class 
    model = model.to(device) 
    
    # Defining the optimizer 
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
    
    # Defining the loss function 
    loss_fn = Loss() 
    
    # Defining the scaler for mixed precision training 
    scaler = torch.amp.GradScaler() 
    
    # Directories for training and validation within the dataset
    root_dir = "/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/PASCAL_dataset"
    train_images_dir = os.path.join(root_dir, "trainval", "img")
    train_ann_dir = os.path.join(root_dir, "trainval", "ann")
    #test_images_dir = os.path.join(root_dir, "test", "img")
    #test_ann_dir = os.path.join(root_dir, "test", "ann")
       
    train_dataset = Dataset(train_images_dir, train_ann_dir, anchors=ANCHORS, transform=train_transforms)
    
    # Defining the train data loader 
    train_loader = torch.utils.data.DataLoader( 
        train_dataset, 
        batch_size = batch_size, 
        num_workers = 2, 
        shuffle = True, 
        pin_memory = True, 
    ) 
    
    # Scaling the anchors 
    scaled_anchors = ( 
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
    ).to(device) 
    
    # Training the model 
    for e in range(1, num_epochs+1): 
        print("Epoch:", e) 
        training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 
    
        # Saving the model 
        if save_model: 
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

# ==========================
# 6. Main Execution
# ==========================
if __name__ == "__main__":
    # Set the root directory where the Ducks dataset is located.
    #dataset_root = "/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/PASCAL_dataset"
    #train_loader, val_loader = create_dataloaders(dataset_root, batch_size=8)
    
    model = ObjectDetectionModel()
    train_model(model, num_epochs=20, learning_rate=ADAM_STEP)
    
    # Save the trained model.
    torch.save(model.state_dict(), "/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/ducks_dataset/duck_object_detection_model.pth")

    #num_classes = n_classes
    #
    ## Creating model and testing output shapes 
    #model = ObjectDetectionModel(num_classes=num_classes) 
    #x = torch.randn((1, 3, IMG_SIZE, IMG_SIZE)) 
    #out = model(x) 
    #print(out[0].shape) 
    #print(out[1].shape) 
    #print(out[2].shape) 
    #
    ## Asserting output shapes 
    #assert model(x)[0].shape == (1, 3, IMG_SIZE//32, IMG_SIZE//32, num_classes + 5) 
    #assert model(x)[1].shape == (1, 3, IMG_SIZE//16, IMG_SIZE//16, num_classes + 5) 
    #assert model(x)[2].shape == (1, 3, IMG_SIZE//8, IMG_SIZE//8, num_classes + 5) 
    #print("Output shapes are correct!")
