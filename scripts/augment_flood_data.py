import random
from pathlib import Path

from PIL import Image, ImageFile
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = Path("data/processed/flood_classified")
TARGET_COUNTS = {
    "mild": 600,
    "severe": 600,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def get_class_dirs(class_name):
    train_dir = DATA_DIR / "train" / class_name
    val_dir = DATA_DIR / "val" / class_name
    if train_dir.exists() and val_dir.exists():
        return [train_dir, val_dir], train_dir

    class_dir = DATA_DIR / class_name
    return [class_dir], class_dir


def get_images(directories):
    images = []
    for directory in directories:
        if not directory.exists():
            continue
        images.extend(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    return sorted(images)


def build_augment(size):
    height, width = size
    blur_kernel = random.choice([3, 5])
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(height, width),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.35),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))],
                p=0.35,
            ),
        ]
    )


def next_aug_path(output_dir, source_stem, index):
    while True:
        candidate = output_dir / f"{source_stem}_aug_{index:03d}.jpg"
        if not candidate.exists():
            return candidate, index + 1
        index += 1


def augment_class(class_name, target_count):
    class_dirs, output_dir = get_class_dirs(class_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_images = [
        path for path in get_images(class_dirs)
        if "_aug_" not in path.stem and not path.stem.startswith("web_")
    ]
    all_images = get_images(class_dirs)
    before_count = len(all_images)

    print(f"{class_name}: before={before_count}, target={target_count}")

    if before_count >= target_count:
        print(f"{class_name}: already meets target, no augmentation needed")
        return

    if not source_images:
        print(f"{class_name}: no source images found, skipping")
        return

    needed = target_count - before_count
    aug_index = 1
    created = 0

    while created < needed:
        img_path = source_images[created % len(source_images)]
        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                transform = build_augment(image.size[::-1])
                augmented = transform(image)
        except Exception as exc:
            print(f"Skipping {img_path}: {exc}")
            created += 1
            continue

        output_path, aug_index = next_aug_path(output_dir, img_path.stem, aug_index)
        augmented.save(output_path, format="JPEG", quality=92)
        created += 1

    after_count = len(get_images(class_dirs))
    print(f"{class_name}: created={created}, after={after_count}")


def main():
    random.seed()
    for class_name, target_count in TARGET_COUNTS.items():
        augment_class(class_name, target_count)


if __name__ == "__main__":
    main()
