# Camera-Calibration
Camera Calibration toolbox using Python &amp; openCV
## Introduction
Cameras have been around in our lives for a while. When first appeared, they were very expensive. Fortunately, in the late 20th century, the pinhole camera was developed and sold at suitable prices such that everybody was able to afford it. However, as is the case with any trade off, this convenience comes at a price. This camera model is typically not good enough for accurate geometrical computations based on images; hence, significant distortions may result in images.

Camera calibration is the process of determining the relation between the camera’s natural units (pixels) and the real world units (for example, millimeters or inches). Technically, camera calibration estimates intrinsic (camera's internal characteristics such as focal length, skew, distortion) and extrinsic (its position and orientation in the world) parameters. Camera calibration is an important step towards getting a highly accurate representation of the real world in the captured images. It helps removing distortions as well.
There are several calibration tools available online, such as:
* [Bouguet Toolbox for Matlab](http://www.vision.caltech.edu/bouguetj/calib_doc/)
* [BoofCV: open source Java library](https://boofcv.org/index.php?title=Tutorial_Camera_Calibration)

Here, I create my own calibration toolbox from scratch using python and OpenCV. I follow Zhang’s description in his paper “A flexible new technique for camera calibration,” with allows calibration from multiple 2D plane images. My toolbox is characterized by its friendly user interface, its compatibility to work on any camera and any operation system, and automation such that instead of taking 2D points on captured images individually and manually, a corner detection algorithm automatically handles this task and saves much time and effort.

## Installation
1. Clone or download this repository.

2. Make sure python 3.x is installed on your PC. To check if and which version is installed, run the following command:
```
python -V
```
If this results an error, this means that python isn’t installed on your PC! please install it from [the original website](https://www.python.org/)

3. (optional) it is recommended that you create a python virtual environment and install the necessary libraries in it to prevent versions collisions:
```
python -m venv CV
```
where CV is the environment name. Once you’ve created a virtual environment, you may activate it.
```
CV\Scripts\activate.bat
```

4. Install required libraries from the provided file (**requirements.txt**):
```
pip install -r requirements.txt
```
Make sure you provide the correct path of **requirements.txt**

5. DONE :) Run the script:
```
python calibration_GUI.py
```

## DEMO
This is the main page of the graphical interface:

![image](https://user-images.githubusercontent.com/9033365/46244812-56756600-c3ed-11e8-9b62-6c9600c025e0.png)

Check the **Detect Corners** checkbox and click **START**. The webcam will start capturing frame and displaying them in the interface:

![image](https://user-images.githubusercontent.com/9033365/46244851-021eb600-c3ee-11e8-8752-054269ff1bbe.png)

If you are happy with the captured image, click **CONFIRM**. Otherwise, click **IGNORE** to take another image. Once you click **CONFIRM** or **IGNORE**, the webcam will continue capturing frames, displaying them, and trying to detect corners. The **Images taken** counter on the top right indicates how many images you have confirmed so far.

![image](https://user-images.githubusercontent.com/9033365/46244963-79a11500-c3ef-11e8-8784-69e840e553aa.png)

At least three images are required. The more, the better the parameters estimation. Once you confirm at least three images, you can click **DONE** on the bottom right. Once clicked:
* the intrinsic and extrinsic parameters will be calculated. Three files will be created: **intrinsic.txt**, **extrinsic.txt**, and **predicted VS actual.txt**
* The 2D points of the first confirmed image will be predicted using the following relation:

![image](https://user-images.githubusercontent.com/9033365/46245137-cede2600-c3f1-11e8-96d5-6e3895f60f1f.png)

* To check accuracy, the predicted 2D points will be displayed on the image in the graphical interface as white circles with a plus sign inside.

![image](https://user-images.githubusercontent.com/9033365/46245120-a1917800-c3f1-11e8-918a-38cdfc13fa97.png)


[Click here to see the demo as video.](https://drive.google.com/file/d/16kSAB0DtYn3Hs7U9yBGAok8P0g7BMp-G/view?usp=sharing)
