import pickle
import cv2
import numpy as np
from mmpose.core.visualization.image import imshow_keypoints
from tqdm import tqdm

# --- Step 1: Load PKL file with keypoints ---
#/data/group1/z40575r/GloFE/tools/openasl_mmpose2/openasl_mmpose/_0Kb5WNSfNo-00-03-12.633-00-03-19.100.pkl # mmpose extracted
#/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/CorrNet_pose_extraction1/_0Kb5WNSfNo-00-03-12.633-00-03-19.100.pkl # CorrNet extracted
#/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/keypoints_pose_256x256_2_best/_0Kb5WNSfNo-00-03-12.633-00-03-19.100.pkl #New extraction

#Another video
# /data/group1/z40575r/GloFE/cropped-vid/XySXQIrwypg-00-00-29.533-00-00-34.966.mp4
#/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/keypoints_pose_256x256_2_best/XySXQIrwypg-00-00-29.533-00-00-34.966.pkl
#/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/keypoints_pose_256x256_1_best/XySXQIrwypg-00-00-29.533-00-00-34.966.pkl
#with open('/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/keypoints_pose_256x256_1_best/XySXQIrwypg-00-00-29.533-00-00-34.966.pkl', 'rb') as f:
# data/group1/z40575r/CorrNet_pose_distillation/CorrNet/keypoints_regression_img2/XySXQIrwypg-00-00-29.533-00-00-34.966.pkl

with open('//data/group1/z40575r/CorrNet_pose_distillation/CorrNet/keypoints_pose_regression_img_v13_a1_b1_40k_ep3/XySXQIrwypg-00-00-29.533-00-00-34.966.pkl', 'rb') as f:
    keypoints = pickle.load(f)  # shape: (195, 133, 3)

if keypoints.shape[-1] == 2:
    conf = np.ones((*keypoints.shape[:-1], 1), dtype=keypoints.dtype)
    keypoints = np.concatenate([keypoints, conf], axis=-1)

#keypoints[..., 0], keypoints[..., 1] = keypoints[..., 1], keypoints[..., 0].copy()
# --- Step 2: Load MP4 file ---
video_path = '/data/group1/z40575r/GloFE/cropped-vid/XySXQIrwypg-00-00-29.533-00-00-34.966.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Step 3: Create a VideoWriter for the output ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('non_sequential_model_1.mp4', fourcc, fps, (width, height))


# --- Optional: Define skeleton and color (you can customize these) ---
# Example for COCO Wholebody
#skeleton = [(0, 1), (1, 2), (2, 3)]  # Replace with actual skeleton
pose_kpt_color = [np.random.randint(0, 255, size=3).tolist() for _ in range(133)]
#pose_link_color = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(skeleton))]


# --- Step 4: Overlay keypoints frame by frame ---
frame_idx = 0
pbar = tqdm(total=min(len(keypoints), frame_count))

scale_x = 224 / 96
scale_y = 224 / 72

while cap.isOpened() and frame_idx < len(keypoints):
    ret, frame = cap.read()
    #print(frame.shape)
    if not ret:
        break

    kpts = keypoints[frame_idx]  # (133, 3)
    
    #rescale
    #kpts[:, 0] *= scale_x
    #kpts[:, 1] *= scale_y
    # Wrap in a list for pose_result
    annotated_frame = imshow_keypoints(
        img=frame,
        pose_result=[kpts],
        #skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        #pose_link_color=pose_link_color,
        kpt_score_thr=0.3,
        radius=3,
        thickness=2
    )

    out.write(annotated_frame)
    frame_idx += 1
    pbar.update(1)

cap.release()
out.release()
pbar.close()
print('✅ Video saved as output_with_keypoints.mp4')
