import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import hashlib

camera_icon_path = "camera.png"
user_folders = [f"user{i}" for i in range(1, 8)]

camera_icon = cv2.imread(camera_icon_path, cv2.IMREAD_GRAYSCALE)
icon_height, icon_width = camera_icon.shape

match_threshold = 0.33

images_with_icon = []
suspected_fake_users = []


def compute_image_hash(image) -> str:
    resized = cv2.resize(image, (100, 100))
    return hashlib.md5(resized).hexdigest()


def has_camera_icon(image, icon) -> bool:
    result = cv2.matchTemplate(image, icon, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= match_threshold


def is_image_blurry(image, threshold=100.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def is_lack_of_variation(image1, image2, threshold=0.9) -> bool:
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity >= threshold


def process_user_folder(user_folder) -> None:
    user_images = os.listdir(user_folder)
    user_flagged_images = set()
    image_hashes = set()
    live_photo_images = []
    image_variations = {}
    duplicate_count = 0
    variation_count = 0

    for image_name in user_images:
        image_path = os.path.join(user_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if gray_image.shape[0] < icon_height or gray_image.shape[1] < icon_width:
            continue

        if has_camera_icon(gray_image, camera_icon):
            images_with_icon.append(image_path)
            user_flagged_images.add(image_name)
            live_photo_images.append(image)

        img_hash = compute_image_hash(image)
        if img_hash in image_hashes:
            duplicate_count += 1
        else:
            image_hashes.add(img_hash)

        if is_image_blurry(image):
            user_flagged_images.add(image_name)

        image_variations[image_name] = gray_image

    for image_name, gray_image in image_variations.items():
        for other_image_name, other_gray_image in image_variations.items():
            if image_name < other_image_name and is_lack_of_variation(
                gray_image, other_gray_image
            ):
                variation_count += 1

    total_images = len(user_images)
    if (
        len(live_photo_images) / total_images > 0.5
        or duplicate_count > 0
        or variation_count > 2
    ):
        suspected_fake_users.append(
            {
                "user": user_folder,
                "flagged_images": list(user_flagged_images),
                "total_images": total_images,
                "duplicate_images": duplicate_count,
                "lack_of_variation": variation_count,
            }
        )


with ThreadPoolExecutor() as executor:
    executor.map(process_user_folder, user_folders)


print("Images with 'camera.png' icon:")
for img in images_with_icon:
    print(img)

for user in suspected_fake_users:
    print(f"\nUser: {user['user']}")
    print(
        f"Flagged {len(user['flagged_images'])} out of {user['total_images']} images:"
    )
    print(f"Duplicate Images: {user['duplicate_images']}")
    print(f"Lack of variation detected in {user['lack_of_variation']} image pairs:")
    for img in user["flagged_images"]:
        print(f" - {img}")
    print(
        "Reason: High proportion of flagged images, duplicates, or lack of variation."
    )
