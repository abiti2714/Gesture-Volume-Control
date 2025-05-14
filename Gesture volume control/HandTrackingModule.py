
import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        The function initializes a hand tracking object with specified parameters using the MediaPipe
        library in Python.

        :param mode: The `mode` parameter is used to specify whether the hand tracking is for static
        images or for video frames. When `mode` is set to `True`, it indicates static image mode, and
        when set to `False`, it indicates video mode, defaults to False (optional)
        :param maxHands: The `maxHands` parameter in the `__init__` method is used to specify the maximum
        number of hands to detect in the image or video frames. By default, it is set to 2, but you can
        adjust this value based on your requirements, defaults to 2 (optional)
        :param detectionCon: The `detectionCon` parameter in the `__init__` method is used to set the
        minimum confidence value for hand detection. This value determines how confident the model needs
        to be in its prediction that a hand is present in the image. If the confidence level falls below
        this threshold, the model
        :param trackCon: The `trackCon` parameter in the `__init__` method is used to set the minimum
        confidence value for hand landmarks to be considered for tracking. This parameter is used in the
        configuration of the `Hands` object from the `mediapipe` library.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)  # Explicitly cast to float
        self.trackCon = float(trackCon)          # Explicitly cast to float

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):  # Indented to be part of the class
        """
        The function `findHands` processes an image to detect and draw landmarks for hands using the
        MediaPipe Hands library in Python.

        :param img: The `img` parameter is the input image that contains the hand(s) you want to detect
        and draw landmarks for
        :param draw: The `draw` parameter in the `findHands` method is a boolean parameter that
        determines whether the detected hand landmarks should be drawn on the image or not. If `draw` is
        set to `True`, the landmarks will be drawn on the image using the `mpDraw.draw_landmarks`
        method, defaults to True (optional)
        :return: The `findHands` method is returning the image `img` with hand landmarks drawn on it if
        the `draw` parameter is set to `True`.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):  # Indented to be part of the class
        """
        The function `findPosition` extracts hand landmarks from an image and returns the landmark
        positions and bounding box coordinates.

        :param img: The `img` parameter in the `findPosition` method is expected to be an image
        (represented as a NumPy array) on which you want to detect and draw landmarks for a hand. This
        image will be processed to identify the landmarks and draw them if the `draw` parameter is set
        to
        :param handNo: The `handNo` parameter in the `findPosition` method is used to specify which hand
        to detect landmarks for. By default, it is set to 0, which means it will detect landmarks for
        the first hand found in the image. You can change this parameter to 1 if you want, defaults to 0
        (optional)
        :param draw: The `draw` parameter in the `findPosition` method is a boolean parameter that
        determines whether to draw the landmarks and bounding box on the image. If `draw` is set to
        `True`, the method will draw circles at the landmark positions and a rectangle around the hand
        region on the image, defaults to True (optional)
        :return: The `findPosition` method returns two values: `self.lmList` and `bbox`. `self.lmList`
        contains a list of landmarks with their corresponding x and y coordinates, while `bbox` contains
        the bounding box coordinates (xmin, ymin, xmax, ymax) of the detected hand in the image.
        """
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):  # Indented to be part of the class
        """
        This function determines the position of fingers based on hand landmarks detected by a hand
        tracking system.
        :return: The `fingersUp` method is returning a list `fingers` that contains binary values (1 or
        0) representing whether each finger is up (1) or down (0). The list includes the thumb and the
        four fingers, with each element corresponding to a finger.
        """
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):  # Indented to be part of the class
        """
        The `findDistance` method calculates the distance between two points on an image and optionally
        draws the points and line connecting them.

        :param p1: The `p1` parameter in the `findDistance` method represents the index of the first
        landmark point in the `lmList` attribute of the class. This index is used to access the x and y
        coordinates of the first landmark point in the `lmList`
        :param p2: The `p2` parameter in the `findDistance` method represents the index of the second
        landmark point in the `lmList` attribute of the class. This method calculates the distance
        between two landmark points (`p1` and `p2`) and returns the length of the line segment connecting
        them
        :param img: The `img` parameter in the `findDistance` method is typically an image (e.g., a frame
        from a video feed) on which you want to draw circles and lines to visualize the distance between
        two points (`p1` and `p2`). The method calculates the distance between the two
        :param draw: The `draw` parameter in the `findDistance` method is a boolean parameter that
        determines whether to draw circles and lines on the image `img` or not. If `draw` is set to
        `True`, the method will draw circles at points `p1` and `p2`, a, defaults to True (optional)
        :return: The `findDistance` method returns three values: 
        1. `length`: The distance between points `p1` and `p2`.
        2. `img`: The image with circles and lines drawn on it if `draw` is set to `True`.
        3. A list containing the coordinates of points `p1`, `p2`, and the midpoint between them.
        """
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame. Check your camera connection.")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(
            img)  # Ensure bbox is also returned
        if len(lmList) >= 5:  # Check if index 4 exists
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
