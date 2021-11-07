import cv2
import itertools
import numpy as np
from time import time
import matplotlib.pyplot as plt
from face_detection import face_detect_box, draw_box_position


vid = cv2.VideoCapture(0)
time1 = 0
while True:
    ret, frame = vid.read()

    face_info = face_detect_box(frame)
    result_frame = draw_box_position(frame, face_info)
    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

    time2 = time()
    if (time2 - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(result_frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2

    cv2.imshow('frame', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()