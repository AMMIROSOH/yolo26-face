import os
import cv2
import numpy as np
import sys
from tqdm import tqdm


def xywh_to_x1y1x2y2(box):
    """
    Convert WIDER format box (x, y, w, h) -> (x1, y1, x2, y2)
    where x,y are top-left, w/h are width/height in pixels.
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return x1, y1, x2, y2


def box_xyxy_to_yolo_norm(size, box_xyxy):
    """
    size: (width, height)
    box_xyxy: (x1, y1, x2, y2) in absolute pixel coords
    returns normalized (cx, cy, w, h) in [0,1] clipped.
    """
    width, height = float(size[0]), float(size[1])
    x1, y1, x2, y2 = box_xyxy
    # clamp to image
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))

    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    nw = bw / width
    nh = bh / height

    # clip final values to [0,1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)

    return cx, cy, nw, nh


def format_yolo_pose_line(numbers):
    """
    Format a YOLO pose label line while preserving numpy floating-point values.
    Class ids and visibility flags remain integers; normalized coordinates remain floats.
    """
    formatted = []
    for i, value in enumerate(numbers):
        if i == 0 or (i >= 5 and (i - 5) % 3 == 2):
            formatted.append(str(int(value)))
        else:
            formatted.append(f"{float(value):.6f}")
    return " ".join(formatted)


def wider2face(root, phase='val', ignore_small=0):
    """
    Parse WIDER label file at {root}/{phase}/label.txt

    Returns dict: {image_abs_path: list_of_boxes}
    Each box is returned as a numpy array [x, y, w, h] (absolute pixels)
    """
    label_path = os.path.join(root, phase, 'label.txt')
    data = {}
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"{label_path} not found")

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() != '']

    current_img = None
    current_size = None

    for line in tqdm(lines, desc=f"Parsing {phase} label.txt"):
        if line.startswith('#'):
            # WIDER's line usually like: "# 0--Parade/0_Parade_marchingband_1_849.jpg"
            parts = line.split()
            # image relative path is usually the last token
            img_rel = parts[-1]
            img_path = os.path.join(root, phase, 'images', img_rel)
            if not os.path.isfile(img_path):
                # try some reasonable alternates (in case path formatting differs)
                alt = os.path.join(root, phase, img_rel)
                if os.path.isfile(alt):
                    img_path = alt
                else:
                    # store path anyway; we will check existence later
                    img_path = os.path.join(root, phase, 'images', img_rel)

            # attempt to read to get dimensions; if fail, current_size stays None and boxes will be stored
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                current_size = (w, h)
            else:
                current_size = None

            current_img = img_path
            data[current_img] = {
                'boxes': [],
                'size': current_size
            }
        else:
            # box line: first 4 numbers are x y w h in absolute pixels
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                box = np.array(parts[0:4], dtype=np.float32)
            except ValueError:
                continue

            # ignore small boxes
            if box[2] < ignore_small or box[3] < ignore_small:
                continue

            # append box (as absolute x,y,w,h)
            if current_img is not None:
                data[current_img]['boxes'].append(box)
            else:
                # malformed file: no image header before boxes, skip
                continue

    return data


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Missing path to WIDERFACE folder.')
        print('Run: python3 val2yolo_pose.py /path/to/original/widerface [/path/to/save/widerface/val]')
        sys.exit(1)
    elif len(sys.argv) > 3:
        print('Too many arguments.')
        print('Run: python3 val2yolo_pose.py /path/to/original/widerface [/path/to/save/widerface/val]')
        sys.exit(1)

    root_path = sys.argv[1]
    label_file = os.path.join(root_path, 'val', 'label.txt')
    if not os.path.isfile(label_file):
        print('Missing label.txt file at', label_file)
        sys.exit(1)

    if len(sys.argv) == 2:
        save_path = os.path.join('widerface', 'val')
    else:
        save_path = sys.argv[2]

    os.makedirs(save_path, exist_ok=True)

    datas = wider2face(root_path, phase='val', ignore_small=0)

    idx = 0
    for img_path, info in tqdm(datas.items(), desc="Writing val images/labels"):
        boxes = info['boxes']
        size = info['size']

        # try to read image, if missing skip with warning
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}, skipping.")
            continue

        h, w = img.shape[:2]
        # prefer actual image size over parsed size
        size = (w, h)

        # prepare label file lines
        label_lines = []
        for box in boxes:
            # box is [x, y, w, h]
            x1, y1, x2, y2 = xywh_to_x1y1x2y2(box)
            cx, cy, nw, nh = box_xyxy_to_yolo_norm(size, (x1, y1, x2, y2))

            # WIDER val typically has no keypoints. Emit 5 keypoints as (0,0,0) (not labeled)
            kps_out = []
            for _ in range(5):
                kps_out.extend([0.0, 0.0, 0])  # x_norm, y_norm, vis=0

            numbers = [0, cx, cy, nw, nh] + kps_out  # class 0 (face)
            line = format_yolo_pose_line(numbers)
            label_lines.append(line)

        # write image and label
        out_img = os.path.join(save_path, f"{idx}.jpg")
        out_txt = os.path.join(save_path, f"{idx}.txt")
        cv2.imwrite(out_img, img)
        with open(out_txt, 'w', encoding='utf-8') as f:
            for l in label_lines:
                f.write(l + "\n")

        idx += 1

    print("Finished. Saved", idx, "images to", save_path)
