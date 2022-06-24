# Import dependencies
import os                                   # For file operations
import cv2                                  # For video rendering
from playsound import playsound             # For playing sounds
from perfect_photo import PerfectPhoto      # To analyze pictures for open eyes and a smile

"""
Takes frames from the webcam, has them analyzed by PerfectPicture class, then saves a picture if thresholds are passed.
Press Q to stop execution.
"""
def main():
    webcam = cv2.VideoCapture(0)                             # Grab webcam
    perf_pic = PerfectPhoto()
    photo_dir = os.path.join(os.getcwd(), "perfect_photos")  # Where to save our photos
    sound_dir = os.path.join(os.getcwd(), "sounds")          # Where our sounds are located
    perf_frames = 0                                          # Consecutive frames with all good expressions
    photo = 0                                                # How many photos we have taken so far. Helps with naming.

    while True:
        _, frame = webcam.read()  # Grab latest frame from the webcam

        # Analyze this frame for faces, their landmarks. Don't render anything on it.
        frame, _, expressions = perf_pic.analyze_img(frame, False, False)

        # If we have a good frame, wait 30 frames (1/2 sec @ 60fps) of consistent good frames to take a pic
        for expression in expressions:
            if expression["eyes_open"] is True and expression["smiling"] is True:
                perf_frames += 1
                if perf_frames >= 30:
                    perf_frames = 0
                    photo_path = os.path.join(photo_dir, str(photo) + ".jpg")
                    saved = cv2.imwrite(photo_path, frame)
                    if saved:
                        photo += 1
                        playsound(os.path.join(sound_dir, "CamShutterHardClick.wav"))  # Play camera shutter sound
                    else:
                        print(f"Saving photo to {photo_path} failed.")                 # TODO: Error handling

        cv2.imshow("Webcam", frame)             # Display the processed frame
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit when "q" is pressed
            break

    # Release webcam, destroy windows
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
