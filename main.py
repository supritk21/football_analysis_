from utils import  read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import numpy as np
import cv2


def main():
    #read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True, 
                                       stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_track(tracks)
    # print(tracks['player'])
   # print(tracks['player'][3]) #{12: {'bbox': [359.51776123046875, 725.4869384765625, 397.4945068359375, 828.58154296875]} 
   #print(tracks['ball'][3]) #{1: {'bbox': [1199.70556640625, 859.2280883789062, 1214.7705078125, 874.965087890625]}}


    #estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement.pkl')

    camera_movement_estimator.add_adjusted_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #track of ball
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"]) 
    
    
    # Speed and distance estimator SpeedAndDistanceEstimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.estimate_speed_and_distance(tracks)
   
    # assign player to team 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['player'][0])

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, player in player_track.items():
            bbox = player['bbox']
            frame = video_frames[frame_num]
            team = team_assigner.get_player_team(frame, bbox, player_id)
            tracks['player'][frame_num][player_id]['team'] = team
            tracks['player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    #assign ball aquation to player
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['player']):
         ball_box = tracks['ball'][frame_num][1]['bbox']
         assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_box)   

         if assigned_player != -1:
             tracks['player'][frame_num][assigned_player]['has_ball'] = True    
             team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
         else:
             team_ball_control.append(team_ball_control[-1])    
    
    team_ball_control = np.array(team_ball_control)


    # Draw output 
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
  
    # Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()





# crop image generation code

    # print(tracks['player'][3]) #
    #save cropped image of a player
    # print(tracks["player"][0].items()) 

    # for track_id, player in tracks['player'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     #crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     #save cropped image
    #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
    #     break
