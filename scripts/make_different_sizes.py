import os
import shutil

def main():
    sizes = [1*16, 2*16, 8*16, 16*16, 256*16, 1024*16]
    input_path = 'data/ngp_images/packed_cifar10_image_fields'
    input_file_paths = os.listdir(input_path)
    input_file_paths.remove('weight_map.json')
    for i in range(len(sizes)):
        current_size_path = os.path.join(f'data/ngp_images/cifar10_size{i}')
        os.makedirs(current_size_path)
        for k in range(sizes[i]):
            src = os.path.join(input_path, input_file_paths[k])
            dst = os.path.join(current_size_path, f'{k}.npy')
            shutil.copy(src, dst)

if __name__ == '__main__':
    main()