import cv2

frame_path = 'data/instant_turntable_render_frame_'
num_frames = 60 * 3
width = 800
height = 800
depth = False

video = cv2.VideoWriter(
    'data/instant_turntable_video.mp4', 
    cv2.VideoWriter_fourcc(*"mp4v"), 
    60, 
    (width, height)
)

for i in range(num_frames):
    if depth:
        current_frame_path = frame_path + str(i) + '_depth' + '.png'
    else:
        current_frame_path = frame_path + str(i) + '.png'
    image = cv2.imread(current_frame_path)
    video.write(image)

video.release()