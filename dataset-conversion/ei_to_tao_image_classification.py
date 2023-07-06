import numpy as np
import argparse, math, shutil, os, json, time, re
from PIL import Image

parser = argparse.ArgumentParser(description='Edge Impulse => TAO Image Classiifcation')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--alpha', type=float, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

image_width, image_height, image_channels = list(X_train.shape[1:])
classes_count = Y_train.shape[1]
classes = [ 'class' + str(n) for n in range(0, classes_count) ]

if image_width != image_height:
    print('ERR: image input size should be square, but was ' +
        str(image_width) + 'x' + str(image_height))
    exit(1)

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with TAO')

def current_ms():
    return round(time.time() * 1000)

total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = current_ms()
converted_images = 0

def convert(X, Y, category):
    global class_count, total_images, zf, last_printed, converted_images

    image_list = []

    images_category_dir = os.path.join(out_dir, 'dataset', category + '_set', category + '_set')
    os.makedirs(images_category_dir, exist_ok=True)

    for ix in range(0, len(X)):
        raw_data = X[ix].copy()

        # un-scale data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        raw_data[:, :, 0] *= std[0]
        raw_data[:, :, 1] *= std[1]
        raw_data[:, :, 2] *= std[2]
        raw_data[:, :, 0] += mean[0]
        raw_data[:, :, 1] += mean[1]
        raw_data[:, :, 2] += mean[2]

        raw_img_data = (np.reshape(raw_data, (image_width, image_height, image_channels)) * 255).astype(np.uint8)
        class_index = np.argmax(Y[ix])
        class_label = classes[class_index]

        images_dir = os.path.join(images_category_dir, class_label)
        os.makedirs(images_dir, exist_ok=True)

        im = Image.fromarray(raw_img_data)
        im.save(os.path.join(images_dir, class_label + '.' + str(ix).zfill(5) + '.jpg'))

        image_list.append(class_label + '/' + class_label + '.' + str(ix).zfill(5) + '.jpg ' + str(class_index))

        converted_images = converted_images + 1
        if (converted_images == 1 or current_ms() - last_printed > 3000):
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = current_ms()

    with open(os.path.join(out_dir, 'dataset', category + '.txt'), 'w') as f:
        f.write('\n'.join(image_list))

convert(X=X_train, Y=Y_train, category='training')
convert(X=X_test, Y=Y_test, category='val')
convert(X=[], Y=[], category='test')

print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')

print('Transforming Edge Impulse data format into something compatible with TAO OK')
print('')

with open(os.path.join(out_dir, 'dataset', 'classes.txt'), 'w') as f:
    f.write('\n'.join(classes))

specs_file = """model_config {
  # Model Architecture can be chosen from:
  # ['resnet', 'vgg', 'googlenet', 'alexnet']
  arch: "mobilenet_v1"
  # for resnet --> n_layers can be [10, 18, 50]
  # for vgg --> n_layers can be [16, 19]
  n_layers: 10
  use_batch_norm: True
  use_bias: False
  all_projections: False
  use_pooling: True
  retain_head: True
  resize_interpolation_method: BICUBIC
  # if you want to use the pretrained model,
  # image size should be "3,224,224"
  # otherwise, it can be "3, X, Y", where X,Y >= 16
  input_image_size: \"""" + f"{image_channels},{image_width},{image_height}" + """\"
}
train_config {
  train_dataset_path: \"""" + os.path.join(args.out_directory, 'dataset/training_set/training_set/') + """\"
  val_dataset_path: \"""" + os.path.join(args.out_directory, 'dataset/val_set/val_set/') + """\"
  # Only ['sgd', 'adam'] are supported for optimizer
  optimizer {
      sgd {
      lr: """ + str(args.learning_rate) + """
      decay: 0.0
      momentum: 0.9
      nesterov: False
      }
  }
  batch_size_per_gpu: 50
  n_epochs: """ + str(args.epochs) + """
  # Number of CPU cores for loading data
  n_workers: 16
  lr_config {
      cosine {
      learning_rate: 0.04
      soft_start: 0.0
      }
  }
  # regularizer
  reg_config {
      # regularizer type can be "L1", "L2" or "None".
      type: "None"
      # if the type is not "None",
      # scope can be either "Conv2D" or "Dense" or both.
      scope: "Conv2D,Dense"
      # 0 < weight decay < 1
      weight_decay: 0.0001
  }
  enable_random_crop: True
  enable_center_crop: True
  enable_color_augmentation: True
  mixup_alpha: 0.2
  label_smoothing: 0.1
  preprocess_mode: "torch"
}
"""

specs_dir = os.path.join(args.out_directory, 'specs')
os.makedirs(specs_dir, exist_ok=True)

with open(os.path.join(specs_dir, 'custom.yaml'), 'w') as f:
    f.write(specs_file)

mobilenet_train_paths = [
    '/usr/local/lib/python3.8/dist-packages/nvidia_tao_tf1/core/templates/mobilenet.py',
    '/home/ubuntu/nvidia_tao_tf1/core/templates/mobilenet.py'
]
found_mobilenet_train_path = False
for mobilenet_train_path in mobilenet_train_paths:
    if os.path.exists(mobilenet_train_path):
        found_mobilenet_train_path = True
        new_lines = []

        with open(mobilenet_train_path, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                m = re.search(r"alpha=([\d\.]+),", line)
                if m:
                    line = line.replace('alpha=' + m[1] + ',', 'alpha=' + str(args.alpha) + ',')
                    new_lines.append(line)
                else:
                    new_lines.append(line)

        with open(mobilenet_train_path, 'w') as f:
            f.write('\n'.join(new_lines))

        break

if not found_mobilenet_train_path:
    print('')
    print('WARN: Could not find mobilenet training template')
    print('')
