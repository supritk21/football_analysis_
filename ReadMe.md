# Football Analysis Project

This project involves analyzing football matches by tracking players and the football, classifying them into teams, capturing their speeds and distances covered, and determining team control over the football.

## Overview

1. **Model Training**:
   - Fine-tuned the YOLOv5 model on a custom Roboflow dataset.
   - The YOLOv5 model detects the positions of all objects in the frame, including players, football, and referees.

2. **Object Tracking**:
   - Tracked players and the football, assigning each object a unique track ID.
   - Calculated the position of the football using the center of its bounding box.
   - Determined player positions by calculating the center of their foot positions.

3. **Speed and Distance Calculation**:
   - Adjusted for camera movement using OpenCV's `calcOpticalFlowPyrLK` and `goodFeaturesToTrack`.
   - Transformed pixel positions to true field positions using perspective transformation.

4. **Performance Metrics**:
   - Calculated the speed and distance covered by each player.
   - Used a frame rate of 24 fps, considering 5 frames to determine player positions and time intervals.

5. **Interpolation**:
   - Interpolated the position of the football when the model failed to track it in some frames.

6. **Team Classification**:
   - Extracted player colors by cropping the top half of the image and separating players from background colors.
   - Used K-means clustering to classify players into two teams based on their colors.

7. **Ball Control**:
   - Assigned ball control to the player closest to the ball, with a specified minimum distance.
   - Calculated team control over the ball by counting ball possession instances for each team.

8. **Output**:
   - Added all calculated features to the output frame for visualization.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- YOLOv5
- Roboflow

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supritk21/football-analysis.git
   cd football-analysis


