import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', type=str, default='data/instant_turntable_render_frame_')
    parser.add_argument('--num_frames', type=int, default=60*3)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--depth', type=bool, default=False)
    args = parser.parse_args()

    video = cv2.VideoWriter(
        'data/instant_turntable_video.mp4', 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        60, 
        (args.width, args.height)
    )

    for i in range(args.num_frames):
        if args.depth:
            current_frame_path = args.frame_path + str(i) + '_depth' + '.png'
        else:
            current_frame_path = args.frame_path + str(i) + '.png'
        image = cv2.imread(current_frame_path)
        video.write(image)

    video.release()