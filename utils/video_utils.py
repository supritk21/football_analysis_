import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames.append(frame)

    return frames


def save_video(frames, output_path):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
            
    