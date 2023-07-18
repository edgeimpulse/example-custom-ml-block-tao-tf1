# NVIDIA TAO example for Edge Impulse (TF1)

This repository contains everything you need to train ML models using the [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) in Edge Impulse. NVIDIA TAO contains a wide range of state-of-the-art ML architectures and transfer learning weights, and these models can be trained in Edge Impulse like any other model. As a primer, read the [Custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) page in the Edge Impulse docs.

This repository covers the TensorFlow 1-based image classification models in the TAO Toolkit ("Image Classification TF1"), there's also an example that covers the [PyTorch-based image classification](https://github.com/edgeimpulse/example-custom-ml-block-tao-pyt) models. To use other models see 'Running other TAO models' below. By default this repository is configured to train using the MobileNetV1 backbone.

> **To see this in action, see [https://www.youtube.com/watch?v=bwquofcF94g](https://www.youtube.com/watch?v=bwquofcF94g)**

## What does this repo do?

This repository runs (orchestrated by [run.sh](run.sh)):

1. It converts the dataset - as passed in by Edge Impulse - into a dataset that can be read by TAO. This happens in [dataset-conversion/ei_to_tao_image_classification.py](dataset-conversion/ei_to_tao_image_classification.py).
2. It writes out a specs file, which instructs TAO on how to train the model. You can find this in [dataset-conversion/ei_to_tao_image_classification.py] from `specs_file = `. In the default configuration we use MobileNetV1, but you have access to all TAO options here.
3. We make a small modification to the TAO codebase to change the alpha of the MobileNet model. Ideally this should be an option exposed by TAO in the specs file, but this is not yet the case.
4. It trains the model using TAO, then copies the ONNX file to the out directory.

If you want to change the backbone or add transfer learning weights, you'll only need to change the specs file.

## Running the pipeline locally

You run this pipeline via Docker if you have an NVIDIA GPU. This encapsulates all dependencies and packages for you; and is great to test out your modifications locally. Or skip to 'Pushing the block back to Edge Impulse' to directly deploy this block in Edge Impulse.

### Running locally via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.19.0 or higher.
3. Create a new Edge Impulse project, and add image data.
4. Under **Create impulse**, set the image width / height to 224, add an 'Image' processing block, and a 'Classification' ML block.
5. Open a command prompt or terminal window.
6. Initialize the block:

    ```
    $ edge-impulse-blocks init

    Edge Impulse Blocks v1.19.3
    ? Choose a type of block: Machine learning block
    ? Choose an option: Create a new block
    ? Enter the name of your block: TAO MobileNetV1 example
    ? Enter the description of your block: TAOv5 example that trains a MobileNetV1 network
    ? What type of data does this model operate on? Image classification
    ? How is your input scaled? PyTorch (scale to 0..1, then normalize using ImageNet mean/std)
    ? Where can your model train? Only on GPU (GPUs are only available for enterprise projects)
    ```

    > **Important:** Use 'PyTorch (scale to 0..1, then normalize using ImageNet mean/std)'

7. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

8. Log in to the NVIDIA container registry to pull TAO containers ([here's how to get an API key](https://docs.nvidia.com/dgx/ngc-registry-for-dgx-user-guide/index.html)).

    ```
    $ docker login nvcr.io
    ```

9. Build the container:

    ```
    $ docker build -t tao-tf1 .
    ```

10. Run the container to test the training pipeline (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --gpus all --rm -v $PWD:/app tao-tf1 --data-directory /app/data --epochs 10 --learning-rate 0.0001 --out-directory out/
    ```

11. This creates one .onnx file in the 'out' directory with your model.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

#### Adding new arguments

To add new arguments to your training pipeline (e.g. to control data augmentation), see [Custom learning blocks > Arguments to your script](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks#arguments-to-your-script).

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.19 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing the block back to Edge Impulse

You can now push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects via **Create impulse > Add new learning block**.

## Running other TAO models

To use a different TAO model you can modify this repository.

* If the model is available in the 'Image Classification (PyT)' or 'Image Classification (TF1)' applications, you just need to change the specs file.
* If your model is avialable in another application, then:
    1. Modify the [Dockerfile](Dockerfile) to pull from the right container.
    2. Modify [dataset-conversion/ei_to_tao_image_classification.py](ei_to_tao_image_classification.py) to do the dataset conversion, and write out a valid specs file.
    3. Modify [run.sh](run.sh) to call the correct TAO runtime commands.
