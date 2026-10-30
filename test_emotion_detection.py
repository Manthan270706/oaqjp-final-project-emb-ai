"""
Unit tests for Emotion Detection module.
"""
import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """
    Test suite for emotion detection function.
    """

    def test_emotion_detector(self):
        """
        Tests the dominant emotion detection across various emotional statements.
        """
        # Statement: I am glad this happened -> dominant emotion: joy
        result_1 = emotion_detector("I am glad this happened")
        self.assertEqual(result_1['dominant_emotion'], 'joy')

        # Statement: I am really mad about this -> dominant emotion: anger
        result_2 = emotion_detector("I am really mad about this")
        self.assertEqual(result_2['dominant_emotion'], 'anger')

        # Statement: I feel disgusted just hearing about this -> dominant emotion: disgust
        result_3 = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result_3['dominant_emotion'], 'disgust')

        # Statement: I am so sad about this -> dominant emotion: sadness
        result_4 = emotion_detector("I am so sad about this")
        self.assertEqual(result_4['dominant_emotion'], 'sadness')

        # Statement: I am really afraid that this will happen -> dominant emotion: fear
        result_5 = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result_5['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()
