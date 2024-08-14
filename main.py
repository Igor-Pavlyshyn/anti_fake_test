import os
import cv2
from concurrent.futures import ThreadPoolExecutor

camera_icon_path = 'camera.png'
user_folders = [f'user{i}' for i in range(1, 8)]


camera_icon = cv2.imread(camera_icon_path, cv2.IMREAD_GRAYSCALE)
icon_height, icon_width = camera_icon.shape

match_threshold = 0.33

images_with_icon = []
suspected_fake_users = []


def has_camera_icon(image_path) -> bool:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return False

    if image.shape[0] < icon_height or image.shape[1] < icon_width:
        return False

    result = cv2.matchTemplate(image, camera_icon, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= match_threshold


def process_user_folder(user_folder) -> None:
    user_images = os.listdir(user_folder)
    user_flagged_images = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda img: (img, has_camera_icon(os.path.join(user_folder, img))), user_images))

    for img_name, has_icon in results:
        if has_icon:
            images_with_icon.append(os.path.join(user_folder, img_name))
            user_flagged_images.append(img_name)

    if len(user_flagged_images) / len(user_images) > 0.5:
        suspected_fake_users.append({
            'user': user_folder,
            'flagged_images': user_flagged_images,
            'total_images': len(user_images)
        })


with ThreadPoolExecutor() as executor:
    executor.map(process_user_folder, user_folders)

print("Images with 'camera.png' icon:")
for img in images_with_icon:
    print(img)

print("\nSuspected Fake Users:")
for user in suspected_fake_users:
    print(f"User: {user['user']}")
    print(f"Flagged {len(user['flagged_images'])} out of {user['total_images']} images:")
    for img in user['flagged_images']:
        print(f" - {img}")
    print("Reason: High proportion of images with the camera icon.")
