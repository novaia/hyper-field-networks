ffmpeg -framerate 5 -pattern_type glob -i 'step*_image0.png' -filter_complex "scale=200:200:flags=neighbor,tile=1x1:nb_frames=1,format=yuv420p" -c:v libx264 -r 60 -pix_fmt yuv420p 0.mp4 &&
ffmpeg -framerate 5 -pattern_type glob -i 'step*_image1.png' -filter_complex "scale=200:200:flags=neighbor,tile=1x1:nb_frames=1,format=yuv420p" -c:v libx264 -r 60 -pix_fmt yuv420p 1.mp4 &&
ffmpeg -framerate 5 -pattern_type glob -i 'step*_image2.png' -filter_complex "scale=200:200:flags=neighbor,tile=1x1:nb_frames=1,format=yuv420p" -c:v libx264 -r 60 -pix_fmt yuv420p 2.mp4 &&
ffmpeg -framerate 5 -pattern_type glob -i 'step*_image3.png' -filter_complex "scale=200:200:flags=neighbor,tile=1x1:nb_frames=1,format=yuv420p" -c:v libx264 -r 60 -pix_fmt yuv420p 3.mp4 &&
ffmpeg -i 0.mp4 -i 1.mp4 -i 2.mp4 -i 3.mp4 -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" -map "[v]" input.mp4 &&
ffmpeg -i input.mp4 -filter_complex "[0]pad=iw:ih+400:0:200:black[v]" -map "[v]" -map 0:a? -c:a copy output.mp4