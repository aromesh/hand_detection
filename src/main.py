import cv2
import mediapipe

#import modules
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

#open cv capture
capture = cv2.VideoCapture(1)

#with statement to free resources after usage
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=2) as hands:

    while (True):

        #capture frame
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #draw landmarks
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
            
        cv2.imshow('Test Hand', frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
capture.release()

