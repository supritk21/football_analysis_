from ultralytics import YOLO

model = YOLO('models/best.pt')  # Load model

results = model.predict('input_videos/08fd33_4.mp4', save = True)  # Inference video

print(results[0])  # Display results

print("=====================================")
for box in results[0].boxes:
    print(box)