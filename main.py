import argparse
import os
import shutil
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection (source argument only)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    args = parser.parse_args()

    model_path = "yolov8n.pt"
    conf = 0.25
    save = True
    task = "detect"

    model = YOLO(model_path)

    results = model.predict(
        source=args.source,
        conf=conf,
        save=save,
        task=task,
        verbose=True
    )

    runs_dir = os.path.join("runs", "detect")
    if os.path.exists(runs_dir):
        latest_run = max(
            [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)],
            key=os.path.getmtime
        )
        latest_image_path = None
        for file in os.listdir(latest_run):
            if file.lower().endswith(('.jpg', '.png')):
                latest_image_path = os.path.join(latest_run, file)
                break

        if latest_image_path:
            shutil.copy(latest_image_path, "latest_result.jpg")
            print(f"\nDetection complete! Latest result saved as: latest_result.jpg")
        else:
            print("\nNo image found in the latest run folder.")
    else:
        print("\nNo runs/detect directory found. Maybe YOLO didn’t save output properly.")

    for r in results:
        print(r)

if __name__ == "__main__":
    main()

