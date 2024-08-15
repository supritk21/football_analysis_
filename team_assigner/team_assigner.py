from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}  #player_id : team_id

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans
    

    def get_player_color(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
        image = frame[y1:y2, x1:x2]
        top_half_img = image[0: int(image.shape[0]/2), :] 

        #get the clustering model
        kmeans = self.get_clustering_model(top_half_img)

        #get labels for each pixels
        labels = kmeans.labels_

        #reshape labels to image shape
        clustered_image = labels.reshape(top_half_img.shape[0],top_half_img.shape[1])
      
        #get the cluster of the corners / non player cluster      
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1-non_player_cluster

        #from player clusetr get center of the player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
         #if player_id is already assigned a team return the team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
       
       
        #get the player color to classify it into teams
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        
        #team_id is 1 or 2 so increment by 1
        team_id = team_id+1
        self.player_team_dict[player_id] = team_id
        if int(player_id) == 129:
            team_id = 1
       
        return team_id