#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --alpha) # e.g. 0.1
      ALPHA="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$ALPHA" ]; then
    echo "Missing --alpha"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

rm -rf /tmp/out

# convert Edge Impulse dataset (in Numpy format)
python3 -u $SCRIPTPATH/dataset-conversion/ei_to_tao_image_classification.py \
    --data-directory $DATA_DIRECTORY \
    --out-directory /tmp/out \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --alpha $ALPHA

# train the model
echo "Training model..."
classification_tf1 train \
    -e /tmp/out/specs/custom.yaml \
    -r /tmp/out/output
echo "Training model OK"
echo ""

# exporting the model to onnx
echo "Exporting model..."
EPOCHS_FORMATTED=$(printf "%03d" $EPOCHS)

classification_tf1 export \
    -e /tmp/out/specs/custom.yaml \
    -m /tmp/out/output/weights/mobilenet_v1_$EPOCHS_FORMATTED.hdf5

echo "Exporting model OK"
echo ""

echo "Copying to output directory..."
cp /tmp/out/output/weights/mobilenet_v1_$EPOCHS_FORMATTED.onnx $OUT_DIRECTORY/model.onnx
echo "Copying to output directory OK"
echo ""
