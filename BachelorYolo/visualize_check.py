import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

PROJECT_DIR = Path(r"C:\Users\abrah\Downloads\BachelorYolo\Project25YoloExport")
IMAGES_DIR = PROJECT_DIR / "images"
LABELS_DIR = PROJECT_DIR / "labels"

# Load class names if you have classes.txt
classes_file = PROJECT_DIR / "classes.txt"
if classes_file.exists():
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
else:
    classes = ["class_0", "class_1", "class_2", "class_3"]  # placeholder

# Pick 6 random images
image_files = list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.jpg"))
samples = random.sample(image_files, min(6, len(image_files)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, img_path in enumerate(samples):
    # Read image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Read corresponding label
    label_path = LABELS_DIR / f"{img_path.stem}.txt"
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h
                
                # Convert to corner coordinates
                x1 = int(x_center - box_w/2)
                y1 = int(y_center - box_h/2)
                x2 = int(x_center + box_w/2)
                y2 = int(y_center + box_h/2)
                
                # Draw rectangle
                color = (255, 0, 0) if class_id == 0 else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label_text = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                cv2.putText(img, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    axes[idx].imshow(img)
    axes[idx].set_title(f"{img_path.name}\n({len(lines) if label_path.exists() else 0} objects)", 
                        fontsize=8)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(PROJECT_DIR / "visual_check.png", dpi=150, bbox_inches='tight')
print(f"\n✅ Visual check saved to: {PROJECT_DIR / 'visual_check.png'}")
print("   Open this image to verify annotations are correct!")
plt.show()