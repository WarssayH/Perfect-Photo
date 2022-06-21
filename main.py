# Import the required dependencies
import cv2                                  # For video rendering
from perfect_picture import PerfectPicture  # To analyze pictures for open eyes and a smile


"""
Takes frames from the webcam, has them analyzed by PerfectPicture class, then displays the analyzed frame. Press Q 
to stop execution.
"""
def main():
    webcam = cv2.VideoCapture(0)
    perf_pic = PerfectPicture(0.22, 0.45)
    while True:
        _, frame = webcam.read()                # Grab latest frame from the webcam

        # Analyze this frame for faces, render landmarks and analysis for open eyes, smile
        frame = perf_pic.analyze_img(frame)

        cv2.imshow("Webcam", frame)             # Display the processed frame
        if cv2.waitKey(16) & 0xFF == ord("q"):  # Quit when "q" is pressed
            break

    # Release webcam, destroy windows
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
