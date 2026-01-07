import os
import csv
# Comprehensive GPU/OpenGL disabling
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
os.environ["GLOG_minloglevel"] = "2"  # Reduce Google logging

from imageio.v2 import mimread, mimwrite

from imageio import get_reader, get_writer
from PIL import Image

import cv2
import numpy as np
import glob
import mediapipe as mp

# Initialize MediaPipe solutions as it is enough for detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# Paths
input_videos_dir = "/netscratch/ctamang/dataset/TED"
output_dir_stage1 = "/netscratch/ctamang/dataset/TED/TED_stage1"
output_dir_stage2 = "/netscratch/ctamang/dataset/TED/TED_stage2"
csv_dir_stage1 = "/netscratch/ctamang/dataset/TED/TED_stage1"

os.makedirs(output_dir_stage1, exist_ok=True)
os.makedirs(output_dir_stage2, exist_ok=True)

# Parameters
MIN_BODY_HEIGHT_RATIO = 0.25
MAX_BODY_HEIGHT_RATIO = 0.75
OUTPUT_SIZE = 512
FRAME_STRIDE = 2

def test(v_p):
    try:
        reader = mimread(v_p)
        print("video read sucessful!")
    except Exception as e:
        print(f"Error reading the video file! Exception: {e}")

def init_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    f = open(csv_path, mode ='w', newline="")
    writer = csv.writer(f)
    writer.writerow([
        "video_name",
        "frame_index",
        "kept",
        "reason",
        "num_hands_detected",
        "face_detected",
        "body_ratio"
    ])
    return f, writer

def stage_1(video_path, output_path, csv_path):
    """Stage 1: Keep frames where both hands and face are visible"""
    
    # Check if supports with print(cv2.getBuildInformation())    
    try:
        reader = get_reader(video_path)
        fps = reader.get_meta_data().get("fps",30)
        video_name = os.path.basename(video_path)
        print(f"video read sucessful! {video_name}")
    except Exception as e:
        print(f"Error reading the video file! Exception: {e}")
    
    writer = get_writer(output_path, fps=fps) 
    
    if fps <= 0:
        fps = 30  # Default FPS if not detectable   
    
    # Initialize models
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    csv_file, csv_writer = init_csv(csv_path)
    frame_count = 0
    kept_frames = 0
     
    try:        
        for idx, frame in enumerate(reader):
            frame_count += 1
            # if idx % FRAME_STRIDE !=0:
            #     continue
            # RGB Default by imageio
            rgb_frame = frame
            h,w, _ = rgb_frame.shape

            # Detect face and hands, mediapipe works with RGB
            results = pose.process(rgb_frame)
            if not results.pose_landmarks:
                reason = "no_pose_detected"
                kept = 0
                csv_writer.writerow([video_name, idx, kept, reason, False, 0, None])
                continue

            # landmarks
            lm = results.pose_landmarks.landmark            
            
            # Check conditions
            face_detected = (
                lm[mp_pose.PoseLandmark.NOSE].visibility > 0.5 and
                lm[mp_pose.PoseLandmark.LEFT_EYE].visibility > 0.5 and
                lm[mp_pose.PoseLandmark.RIGHT_EYE].visibility > 0.5
            )
            both_hands_detected = (
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5 and
                lm[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5 
            )                
            
            if face_detected == 0 or both_hands_detected == 0:
                kept = 0
                reason = "face_hands_not_detected"               
            
            elif both_hands_detected and face_detected:
                # kept = 1
                # reason = "face_hands_detected"              

                # Distance filtering (ignore too close / too far )                       
                shoulder_y = (
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                ) / 2         
                
                hip_y = (
                    lm[mp_pose.PoseLandmark.LEFT_HIP].y +
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y
                ) / 2 
                
                body_ratio = abs(hip_y- shoulder_y)
                if not (MIN_BODY_HEIGHT_RATIO < body_ratio < MAX_BODY_HEIGHT_RATIO):
                    kept = 0
                    reason= "body_too_close_or_far"
                else:
                    kept = 1
                    reason= "face_hands_detected_properly"
            

            if kept:
                writer.append_data(rgb_frame)
                kept_frames += 1 
               
            csv_writer.writerow([
                video_name,
                idx,
                kept,
                reason,
                face_detected,
                both_hands_detected,
                body_ratio
            ])
    except Exception as e:
        print(f"Error in stage_1: {e}")
    finally:
        writer.close()
        pose.close()
        reader.close()
        csv_file.close()
        
    print(f"Stage 1: {kept_frames}/{frame_count} frames kept -> {output_path}")
    return kept_frames

def stage_2(video_path, output_path):
    # cap = cv2.VideoCapture(video_path)
    reader = get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    writer = get_writer(output_path, fps=fps)

    face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    kept_count = 0
    frame_idx = 0
 
    for frame in reader:
        frame_idx += 1
        rgb = frame
        h, w, _ = frame.shape        
        
        # Detect face and hands using Tasks API
        face_result = face_mesh.process(rgb)
        hand_result = hands.process(rgb)

        points = []

        # Face landmarks
        if face_result.multi_face_landmarks:
            for lm in face_result.multi_face_landmarks[0].landmark:
                points.append([int(lm.x * w), int(lm.y * h)])

        # Hand landmarks
        if hand_result.multi_hand_landmarks:
            for hand in hand_result.multi_hand_landmarks:
                for lm in hand.landmark:
                    points.append([int(lm.x * w), int(lm.y * h)])

        if len(points) == 0:
            continue

        points = np.array(points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # Expand bounding box (vertical)
        box_h = y_max - y_min
        y_min = max(0, y_min - int(0.5 * box_h))
        y_max = min(h, y_max + int(0.2 * box_h))

        # clamp all coordinate so it doesn't exceed image boundary
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # skip invalid crops
        if x_max <= x_min or y_max <= y_min:
            continue     
                
        crop = rgb[y_min:y_max, x_min:x_max]
        
        #resize to 512*512
        pil_img= Image.fromarray(crop)
        pil_img = pil_img.resize((OUTPUT_SIZE,OUTPUT_SIZE), Image.Resampling.LANCZOS)

        # if out is None:
        #     out = cv2.VideoWriter(output_path, fourcc, fps, (OUTPUT_SIZE, OUTPUT_SIZE))
      
        # out.write(crop_resized)
        writer.append_data(np.array(pil_img))
        kept_count += 1

    # cap.release()
    face_mesh.close()
    hands.close()
    writer.close()
    
    print(f"Stage 2 complete: kept frames {kept_count}/{frame_idx}.")

def process_videos():
    video_files = glob.glob(os.path.join(input_videos_dir, "*.mp4"))
    print(f"Found {len(video_files)} videos.")
    
    successful_videos = 0
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"\n[{i+1}/{len(video_files)}] Processing {video_name}")
        
        stage1_path = os.path.join(output_dir_stage1, f"new_1_{video_name}")
        stage2_path = os.path.join(output_dir_stage2, f"new_stage2_{video_name}")
        # success = test(video_path)
        csv_path = os.path.join(csv_dir_stage1, f"csv_{video_name}")
        # Run stage 1
        stage1_success = stage_1(video_path, stage1_path, csv_path)
        
        # Only run stage 2 if stage 1 was successful and produced output
        # if stage1_success and os.path.exists(stage1_path) and os.path.getsize(stage1_path) > 1024:  # At least 1KB
        #     stage_2(stage1_path, stage2_path)
        #     successful_videos += 1
        # else:
        #     print(f"Skipping stage 2 for {video_name} - insufficient output from stage 1")
    
    print(f"\nCompleted! Successfully processed {successful_videos}/{len(video_files)} videos")

if __name__ == "__main__":
    process_videos()
