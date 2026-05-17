import cv2
import os
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_file, fps):
    # Get list of images in the folder, sorted by name
    images = []
    for i in tqdm(range(1, 100000000)):
        name = f"{i}.png"
        # name = f"{i}.jpg"
        if name in os.listdir(image_folder):
            images.append(name)  # Sort images alphabetically, assuming they are named in sequential order
        else:
            # continue
            break

    # Check if there are any images in the folder
    if not images:
        print("No images found in the specified folder.")
        return

    print(f"A total of {len(images)} images")
    # Read the first image to get the size (width, height)
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Loop through all images and add them to the video
    for idx, image_name in enumerate(images):
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error reading image: {image_path}")
            continue

        video.write(frame)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} images...")

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video_file}")

# Example usage:
# image_folder = './output/TG_N32_dilute_nv32_VF25_3layers_rbf_upsample2'  # Replace with the path to your images folder
# output_video_file = 'TG_N32_dilute_nv32_VF25_3layers_rbf_upsample2.mp4'  # The name of the output video file
# image_folder = './output/parabolic_shape8_nv2000_VF30_auglag'  # Replace with the path to your images folder
# output_video_file = 'parabolic_shape8_nv2000_VF30_auglag.mp4'  # The name of the output video file
# image_folder = './output/parabolic_25fall'  # Replace with the path to your images folder
# output_video_file = 'parabolic_shape8_nv2000_VF30_noboxes.mp4'  # The name of the output video file
# image_folder = './output/TG_N32_diluteVF12_nv2220'  # Replace with the path to your images folder
# image_folder = './output/linshi'  # Replace with the path to your images folder
# output_video_file = 'linshi_debug_modecut_c0.3_32ksteps.mp4'  # The name of the output video file
# image_folder = './output/job239359'  # Replace with the path to your images folder
# output_video_file = 'linshi_debug_modecut12_c0.3_repulse1e5_eta2.25.mp4'
# image_folder = "./output/vesnet_N128"
# output_video_file = 'linshi_vesnet_N128.mp4'  # The name of the output video file
# image_folder = "./output/linshi_biem_N128"
# output_video_file = 'shan_BIEM_N128_with_35k.mp4'  # The name of the output video file
# image_folder = './output/shan_BIEM_N128_without'  # Replace with the path to your images folder
# output_video_file = 'shan_BIEM_N128_without.mp4'  # The name of the output video file
# image_folder = './output/does_near_help_without'  # Replace with the path to your images folder
# output_video_file = 'vesnet_N128_does_near_help_without.mp4'  # The name of the output video file

# image_folder = './output/GNN_vesnet_single_AUG10'  # Replace with the path to your images folder
# output_video_file = 'Vesnet_single_AUG10.mov'  # The name of the output video file
Idx = 2360
image_folder = f'./output/GNN_normal_size_sampleW_{Idx}'  # Replace with the path to your images folder
output_video_file = f'GNN_normal_size_sampleW_{Idx}.mov'  # The name of the output video file

# for fileIdx in range(1, 12):
#     image_folder = f'./output/GNN_training_single{fileIdx}'  # Replace with the path to your images folder
#     output_video_file = f'GNN_training_single{fileIdx}.mov'  # The name of the output video file
# image_folder = './output/sep1_parabolic_moving_window'  # Replace with the path to your images folder
# output_video_file = 'parabolic_25fall_moving_camera.mp4'  # The name of the output video file



fps = 60  # Frames per second

create_video_from_images(image_folder, output_video_file, fps)
