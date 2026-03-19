import os
import torch
import cv2

class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir):
        self.imgs = os.listdir(img_dir)
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img/255.0, dtype=torch.float32).permute(2,0,1)

        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, xc, yc, bw, bh = map(float, line.split())

                xmin = (xc - bw/2) * w
                ymin = (yc - bh/2) * h
                xmax = (xc + bw/2) * w
                ymax = (yc + bh/2) * h

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # drone

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return img, target

    def __len__(self):
        return len(self.imgs)