import os
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from noise import NoiseImage

MAX_WORKERS = None


def load_digit_images(digit_image_dir, key):
    img_path = os.path.join(digit_image_dir, f"digit_{key}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_color


def create_number_image(digit_images):
    # 各数字を並べて一つの画像に結合します
    digit_imgs = []
    bounding_boxes = []
    x_offset = 0
    max_height = 0
    for img in digit_images:
        h, w, _ = img.shape
        max_height = max(max_height, h)
        digit_imgs.append((img, (w, h)))

    total_width = sum([w for img, (w, h) in digit_imgs])
    # 数字間のランダムなマージンを計算
    margin = random.randint(0, 15)
    total_margin = sum(margin for _ in range(len(digit_imgs) - 1))
    total_width += total_margin

    # カラー画像として初期化
    number_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    pad = 10
    for i, (img, (w, h)) in enumerate(digit_imgs):
        y_offset = (max_height - h) // 2
        x_offset = sum([w + margin - pad for img, (w, h) in digit_imgs[:i]])
        number_image[y_offset : y_offset + h, x_offset : x_offset + w] = img
        bbox = [x_offset + pad, y_offset + pad, x_offset + w - pad, y_offset + h - pad]
        bounding_boxes.append(bbox)

    return number_image, bounding_boxes


def apply_random_transform(image, bounding_boxes):
    # 画像に余白を追加しておく
    border = int(max(image.shape[0], image.shape[1]) * 0.3)
    padded_image = cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    padded_rows, padded_cols, _ = padded_image.shape

    # アスペクト比の変更を適用（横方向のスケーリング）
    aspect_ratio = random.uniform(0.8, 1.2)
    new_cols = int(padded_cols * aspect_ratio)
    padded_image = cv2.resize(
        padded_image, (new_cols, padded_rows), interpolation=cv2.INTER_LINEAR
    )
    bounding_boxes = [
        [
            int((bbox[0] + border) * aspect_ratio),
            int((bbox[1] + border)),
            int((bbox[2] + border) * aspect_ratio),
            int((bbox[3] + border)),
        ]
        for bbox in bounding_boxes
    ]
    padded_cols = new_cols

    # ランダムな回転角度、スケーリング、平行移動
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.5, 1.2)
    tx = random.uniform(-padded_cols * 0.05, padded_cols * 0.05)
    ty = random.uniform(-padded_rows * 0.05, padded_rows * 0.05)

    # アフィン変換行列を計算
    M_affine = cv2.getRotationMatrix2D((padded_cols / 2, padded_rows / 2), angle, scale)
    M_affine[0, 2] += tx
    M_affine[1, 2] += ty

    # パースペクティブ変換のための点を計算
    dst_size = (padded_cols, padded_rows)
    src_pts = np.float32(
        [
            [0, 0],
            [padded_cols - 1, 0],
            [padded_cols - 1, padded_rows - 1],
            [0, padded_rows - 1],
        ]
    )
    max_displacement = 0.1 * min(padded_cols, padded_rows)
    dst_pts = src_pts + np.float32(
        [
            [
                random.uniform(-max_displacement, max_displacement),
                random.uniform(-max_displacement, max_displacement),
            ]
            for _ in range(4)
        ]
    )

    # アフィン変換行列を3x3に拡張
    M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])

    # パースペクティブ変換行列を計算
    M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 全体の変換行列（ホモグラフィー行列）を計算
    M_total = M_perspective @ M_affine_3x3

    # 画像に全体の変換を適用
    transformed_image = cv2.warpPerspective(
        padded_image,
        M_total,
        dst_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0],
    )

    # バウンディングボックスを変換
    new_bounding_boxes = []
    for bbox in bounding_boxes:
        bbox_pts = np.float32(
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]
        ).reshape(-1, 1, 2)
        # 全体の変換行列を適用
        transformed_pts = cv2.perspectiveTransform(bbox_pts, M_total)
        transformed_pts = transformed_pts.reshape(-1, 2)
        x_coords = transformed_pts[:, 0]
        y_coords = transformed_pts[:, 1]
        new_bbox = [
            int(np.min(x_coords)),
            int(np.min(y_coords)),
            int(np.max(x_coords)),
            int(np.max(y_coords)),
        ]
        new_bounding_boxes.append(new_bbox)

    return transformed_image, new_bounding_boxes


def change_background_and_digit_colors(image):
    # 画像をグレースケールに変換してマスクを作成
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 数字の色と背景色をランダムに設定（BGR）
    digit_color = [random.randint(0, 255) for _ in range(3)]

    # 数字の部分を着色
    digit_colored = np.full(image.shape, digit_color, dtype=np.uint8)
    return digit_colored, mask_inv


def place_on_canvas(image, canvas, mask_inv):
    img_h, img_w, _ = image.shape
    scale = 1.0  # スケーリングファクターの初期化

    x_offset = random.randint(0, canvas.shape[1] - img_w)
    y_offset = random.randint(0, canvas.shape[0] - img_h)

    a = cv2.bitwise_and(
        canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w],
        canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w],
        mask=mask_inv,
    )

    b = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask_inv))

    canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w] = a + b

    return canvas, x_offset, y_offset, img_w, img_h, scale


def update_bounding_boxes(bounding_boxes, x_offset, y_offset, scale=1.0):
    updated_bboxes = []
    for bbox in bounding_boxes:
        new_bbox = [
            int(bbox[0] * scale + x_offset),
            int(bbox[1] * scale + y_offset),
            int(bbox[2] * scale + x_offset),
            int(bbox[3] * scale + y_offset),
        ]
        updated_bboxes.append(new_bbox)
    return updated_bboxes


def add_gradient_noise(image):
    h, w = image.shape[:2]

    if random.choice([True, False]):
        gradient = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    else:
        gradient = np.tile(np.linspace(0, 1, h, dtype=np.float32), (w, 1)).T

    # ランダムな強度と方向を設定
    intensity = random.uniform(0, 1)
    gradient = gradient * intensity

    # グラデーションをカラー画像に拡張
    gradient = cv2.merge([gradient, gradient, gradient])

    # 画像にグラデーションを適用
    image_with_gradient = cv2.convertScaleAbs(image.astype(np.float32) * (1 + gradient))

    return image_with_gradient


def save_image_and_annotations(image, bboxes, filename_prefix, output_dir, labels):
    # 画像を保存
    image_path = os.path.join(output_dir, "JPEGImages", f"{filename_prefix}.png")
    cv2.imwrite(image_path, image)

    # XMLアノテーションを作成
    annotation = ET.Element("annotation", verified="yes")
    ET.SubElement(annotation, "folder").text = "Annotation"
    ET.SubElement(annotation, "filename").text = f"{filename_prefix}.png"
    ET.SubElement(annotation, "path").text = image_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image.shape[1])
    ET.SubElement(size, "height").text = str(image.shape[0])
    ET.SubElement(size, "depth").text = str(
        image.shape[2] if len(image.shape) > 2 else 1
    )
    ET.SubElement(annotation, "segmented").text = "0"

    for bbox, label in zip(bboxes, labels):
        xmin = int(max(0, bbox[0]))
        ymin = int(max(0, bbox[1]))
        xmax = int(min(image.shape[1], bbox[2]))
        ymax = int(min(image.shape[0], bbox[3]))

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    # XMLファイルを保存
    xml_path = os.path.join(output_dir, "Annotations", f"{filename_prefix}.xml")
    tree = ET.ElementTree(annotation)
    tree.write(xml_path)


def process_image(i):
    output_dir = "output_images"
    digit_image_dir = "output_digits"

    num_digits = random.randint(1, 3)
    labels = [str(random.randint(0, 9)) for _ in range(num_digits)]
    digits_list = [load_digit_images(digit_image_dir, label) for label in labels]
    number_image, bounding_boxes = create_number_image(digits_list)
    transformed_image, new_bounding_boxes = apply_random_transform(
        number_image, bounding_boxes
    )
    colored_image, mask_inv = change_background_and_digit_colors(transformed_image)

    gradient_image = add_gradient_noise(colored_image)

    canvas_image, x_offset, y_offset, img_w, img_h, scale = place_on_canvas(
        gradient_image,
        NoiseImage().generate(N=224 * 3, count=5).astype(np.uint8),
        mask_inv,
    )

    final_bounding_boxes = update_bounding_boxes(
        new_bounding_boxes, x_offset, y_offset, scale
    )
    filename_prefix = f"augmented_{i}"
    save_image_and_annotations(
        canvas_image, final_bounding_boxes, filename_prefix, output_dir, labels
    )
    return filename_prefix


def main():
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ImageSets", "Main"), exist_ok=True)
    Path.touch(Path(os.path.join(output_dir, "ImageSets", "Main", "test.txt")))
    Path.touch(Path(os.path.join(output_dir, "ImageSets", "Main", "trainval.txt")))
    Path.touch(Path(os.path.join(output_dir, "labels.txt")))

    num_samples = 1000
    test_num = 50
    filelist = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        filelist = list(
            tqdm(
                executor.map(process_image, range(num_samples + test_num)),
                total=num_samples + test_num,
                desc="Files",
                leave=False,
            )
        )

    with open(os.path.join(output_dir, "ImageSets", "Main", "trainval.txt"), "w") as f:
        f.write("\n".join(filelist[:num_samples]))

    with open(os.path.join(output_dir, "ImageSets", "Main", "test.txt"), "w") as f:
        f.write("\n".join(filelist[num_samples:]))

    with open(os.path.join(output_dir, "labels.txt"), "w") as f:
        f.write("\n".join([str(i) for i in range(10)]))


if __name__ == "__main__":
    main()
