import unittest  # For unit testing
import cv2       # For video rendering
import os        # For file operations

# NON_FUNCTIONAL
# WIP

from perfect_photo import PerfectPhoto

"""Test PerfectPhoto on images within a given directory"""
class TestImages(unittest.TestCase):
    def __init__(self, true_dir, false_dir):
        self.true_dir = os.getcwd()
        self.false_dir = os.getcwd()
        self.test = PerfectPhoto()

    def test_single_face(self):
        # TODO: Read all images from storage directory, analyze them with perfect_photo
        for img in os.listdir(self.img_dir):
            image = cv2.imread(img)
            analyzed, features, expressions = self.test.analyze_img(image)

            # TODO: Render results on image and display, verify test results
            self.assertEqual(len(features), 1)
            self.assertListEqual(expressions, [{"eyes_open": True, "smiling": True}])

            # Show test images
            cv2.imshow("Test Image " + f"'{self.img_dir}'", analyzed)  # Display the processed image
            cv2.waitKey(0)

        cv2.destroyAllWindows()  # Destroy windows

    def test_multi_face(self):
        test = PerfectPhoto(self.img_dir)
        cv2.imshow("Tested Image " + f"'{self.img_dir}'", test)  # Display the processed frames


"""Test PerfectPhoto on videos within a given directory"""
class TestVideo(unittest.TestCase):
    def __init__(self, dir):
        self.img_dir = dir

    def test_single_face(self):
        test = PerfectPhoto()

    def test_multi_face(self):
        test = PerfectPhoto()


"""Test PerfectPhoto on a live feed from a given webcam"""
class TestWebcam(unittest.TestCase):
    def __init__(self, dir):
        self.img_dir = dir

    def test_single_face(self):
        test = PerfectPhoto()

    def test_multi_face(self):
        test = PerfectPhoto()