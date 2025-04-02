# Eye-Tracking-ViT

## calib.html
Gaze Game: Click the Dot!

This is a simple web-based application designed to collect eye-tracking data through a camera feed. Users are prompted to click on red dots appearing randomly across the screen. Each click triggers a snapshot from the webcam and logs the (x, y) target.

OS: 14inch MAC Laptop recommended

### How to Use?
1. Double-click calib.html or run it from a local server.
2. Create a folder of your name inside data.
2. Choose the newly created folder where captured images and data will be saved.
3. Start a game, and the app will go full-screen and access your webcam.
4. Click the red dot.
5. Repeat until all 16 points are completed.

All saved in the folder you selected:
    capture_data.json: JSON log with coordinates and image filenames
    image{id}.jpg: Webcam snapshot for each point (e.g., image1.jpg)

Please organize the files like (if you play 3 times):
    ![Alt text for the image](https://github.com/harryshu1997/Eye-Tracking-ViT/blob/main/tree.png)
            
