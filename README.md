# Celebrity Lookalike Differentiation Using YOLOv5

This repository contains a project that uses YOLOv5 and PyTorch to detect and differentiate between images of two celebrities having similar facial features. The project leverages YOLOv5 for object detection and image classification tasks, specifically designed to identify and distinguish between celebrities' faces.

## Project Overview

This project demonstrates the ability to detect, classify, and differentiate between facial images of two celebrities. The YOLOv5 model, pre-trained on a wide range of images, is fine-tuned using a custom dataset containing images of two celebrities.

### Steps Involved:
1. **Setup YOLOv5 with PyTorch**: We use the YOLOv5 model for detecting and classifying objects (in this case, celebrities' faces) from images.
2. **Prepare a Dataset**: Custom images of celebrities are labeled and trained to help the model distinguish between them.
3. **Train the Model**: Fine-tune the YOLOv5 model with the custom dataset.
4. **Detect Celebrities**: Use the trained model to detect and classify new images.

## Requirements

- Python 3.x
- PyTorch
- YOLOv5
- OpenCV or os module
- LabelImg (for labeling images)
- Matplotlib
- Other dependencies listed in `requirements.txt`

### Installation Instructions:

1. Clone the repository for YOLOv5:
    ```bash
    git clone https://github.com/ultralytics/yolov5
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r yolov5/requirements.txt
    pip install backports.tarfile>=1.2
    pip install pyqt5 lxml --upgrade
    ```

3. Clone LabelImg for image labeling:
    ```bash
    git clone https://github.com/HumanSignal/labelImg
    ```

4. Label your dataset images using `labelImg` to create custom annotations.

### Download YOLOv5 and Pre-trained Weights:

- YOLOv5 repository: [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- Pre-trained weights: [YOLOv5s weights](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
- LabelImg for labeling images: [LabelImg GitHub](https://github.com/tzutalin/labelImg)

## Usage

1. **Download and Prepare Dataset**:
    - Place your labeled images in a directory and prepare a `dataset.yaml` file with paths and labels for training.
    - Use `LabelImg` to annotate images, saving them in YOLO format.

2. **Train the Model**:
    To train the YOLOv5 model with your custom dataset, use the following command:
    ```bash
    cd yolov5
    python train.py --img 320 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --workers 2
    ```

3. **Test the Model**:
    Once the model is trained, use it for inference on new images:
    ```python
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # Load pre-trained model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform detection on an image
    img = 'path_to_image.jpg'
    results = model(img)

    # Display the results
    plt.imshow(np.squeeze(results.render()))
    plt.show()
    ```

## Results

Once the model is trained, you can use it to differentiate between the two celebrities based on facial features and other unique attributes detected by YOLOv5.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [PyTorch](https://pytorch.org/)
