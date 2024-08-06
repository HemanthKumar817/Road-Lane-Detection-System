#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Function to mask the region of interest
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# Function to detect lane lines
def detect_lane_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return line_image

# Function to process and display video using Matplotlib
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Initialize the plot
    fig, ax = plt.subplots()
    plt.ion()  # Enable interactive mode for real-time updates
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lane lines in the frame
        result = detect_lane_lines(frame)
        
        # Convert the BGR image to RGB for display with Matplotlib
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Clear the previous frame and update with the new frame
        ax.clear()
        ax.imshow(result_rgb)
        ax.axis('off')  # Hide axes
        plt.draw()
        plt.pause(0.01)  # Pause to simulate real-time video display
    
    cap.release()
    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the final plot

# Function to handle file opening
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4"), ("Image Files", "*.jpg;*.png")])
    if file_path.endswith('.mp4'):
        process_video(file_path)
    else:
        image = cv2.imread(file_path)
        result = detect_lane_lines(image)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Main function to create the Tkinter GUI
def main():
    root = tk.Tk()
    root.title("Lane Detection System")
    
    button = tk.Button(root, text="Open File", command=open_file)
    button.pack()
    
    root.mainloop()

# Run the main function to start the GUI
if __name__ == "__main__":
    main()
