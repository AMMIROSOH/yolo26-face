import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data


class WiderFaceDetection(data.Dataset):
    """
    Lightweight parser for WIDER-like label.txt files where images are introduced by lines
    starting with '# ' followed by one or more lines containing floats per face:
      x y w h [optional landmarks/other values...]

    This class normalizes the label parsing but does NOT assume an exact number of extra
    columns. It will take up to 15 following floats as 5 keypoint pairs (x,y,v) v stands for visibility and if
    keypoints are missing they are filled with -1.
    """

    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []  # list of lists of face label lists

        # read file
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip() for ln in f]

        labels = []
        is_first = True
        base_dir = os.path.dirname(txt_path)

        for line in lines:
            if not line:
                continue
            if line.startswith('#'):
                # start of a new image entry
                if not is_first:
                    # push previous image labels
                    self.words.append(labels.copy())
                    labels.clear()
                else:
                    is_first = False

                # line after '# ' is relative path in WIDER -> make absolute
                img_rel = line[1:].strip()  # drop '#' and spaces
                # try to resolve common cases: if label file stored in same dir as images or parent
                # We'll try a couple of locations later when reading the image.
                img_path = os.path.join(os.path.dirname(txt_path), 'images', img_rel)
                self.imgs_path.append(img_path)
            else:
                # label line: floats separated by spaces
                parts = [p for p in line.split() if p != '']
                try:
                    nums = [float(x) for x in parts]
                except:
                    # skip malformed lines
                    continue

                # Expect at least 4 values: x y w h
                if len(nums) < 4:
                    continue

                # Build a standardized label: [x, y, w, h, kp1x, kp1y, ..., kp5x, kp5y]
                # take next up to 15 floats as kps; pad with -1 if missing
                kps = nums[4:4 + 15]
                if len(kps) < 15:
                    kps = kps + [-1.0] * (15 - len(kps))

                standard = [nums[0], nums[1], nums[2], nums[3]] + kps
                labels.append(standard)

        # push last image's labels
        self.words.append(labels.copy())

    def __len__(self):
        return len(self.imgs_path)

def detection_collate(batch):
    """
    Collate function that expects each element of batch to be (img_tensor, target_tensor)
    Returns (batched_images, list_of_targets) where list_of_targets[i] is target tensor for image i.
    """
    imgs = []
    targets = []
    for sample in batch:
        img, tgt = sample
        imgs.append(img)
        targets.append(tgt)
    # stack images along 0
    imgs = torch.stack(imgs, 0)
    return imgs, targets


def convert_wider_to_yolo_pose(label_txt_path, save_path):
    """
    Standalone converter: reads label_txt_path and writes image files + YOLO pose labels
    into save_path. Output label files follow Ultralytics pose format:
      per-object line: class cx cy w h kp1_x kp1_y kp1_vis ... kp5_x kp5_y kp5_vis
    """
    if not os.path.isfile(label_txt_path):
        raise FileNotFoundError(f"{label_txt_path} not found")

    os.makedirs(save_path, exist_ok=True)

    ds = WiderFaceDetection(label_txt_path)  # we reuse parser to get structured labels

    for idx in range(len(ds)):
        img_path = ds.imgs_path[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}, skipping")
            continue

        height, width, _ = img.shape
        labels = ds.words[idx]  # standardized parsed labels

        if len(labels) == 0:
            # you may want to still copy image, but no label file
            dst_img = os.path.join(save_path, os.path.basename(img_path))
            cv2.imwrite(dst_img, img)
            continue

        txt_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        txt_path = os.path.join(save_path, txt_name)

        with open(txt_path, 'w', encoding='utf-8') as f:
            for label in labels:
                x, y, w, h = label[0], label[1], label[2], label[3]
                # clip
                x = max(0.0, x)
                y = max(0.0, y)
                w = max(0.0, w)
                h = max(0.0, h)

                cx = (x + w / 2.0) / float(width)
                cy = (y + h / 2.0) / float(height)
                nw = w / float(width)
                nh = h / float(height)

                raw_kps = label[4:]
                kps_out = []
                for i in range(0, 15, 3):
                    kx = raw_kps[i]
                    ky = raw_kps[i + 1]
                    kv = raw_kps[i + 2]
                    if kx < 0 or ky < 0:
                        kps_out.extend([0.0, 0.0, 0])
                    else:
                        nkx = min(max(kx / float(width), 0.0), 1.0)
                        nky = min(max(ky / float(height), 0.0), 1.0)
                        # k visibility in wider face is (0, 1) but in yolo its (0, 1, 2)
                        kps_out.extend([nkx, nky, int(kv) + 1])

                # compose line: 1 class + 4 bbox + 15 kp entries = 20 numbers
                numbers = [0, cx, cy, nw, nh] + kps_out
                line = " ".join(f"{x:.6f}" if isinstance(x, float) else str(int(x)) for x in numbers)
                f.write(line + "\n")

        # copy image
        dst_img = os.path.join(save_path, os.path.basename(img_path))
        cv2.imwrite(dst_img, img)
        print(f"Saved: {dst_img}, {txt_path}")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Missing path to WIDERFACE train folder.')
        print('Run command: python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]')
        exit(1)
    elif len(sys.argv) > 3:
        print('Too many arguments were provided.')
        print('Run command: python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]')
        exit(1)

    original_path = sys.argv[1]
    if len(sys.argv) == 2:
        save_path = os.path.join('widerface', 'train')
    else:
        save_path = sys.argv[2]

    os.makedirs(save_path, exist_ok=True)

    label_txt = os.path.join(original_path, 'label.txt')
    if not os.path.isfile(label_txt):
        print('Missing label.txt file.')
        exit(1)

    convert_wider_to_yolo_pose(label_txt, save_path)
    print("Conversion finished.")