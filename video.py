from moviepy.editor import ImageSequenceClip
import argparse
import imageio

def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + 'output_video.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    imgs = [args.image_folder+"movie"+str(i*10)+".png" for i in range(36)]
    clip = ImageSequenceClip(imgs, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':

    main()
