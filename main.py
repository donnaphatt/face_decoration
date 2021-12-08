import cv2

import numpy as np
from time import time
import matplotlib.pyplot as plt
from face_detection import detectFacialLandmarks, isOpen, overlay, mp_face_mesh

eye = cv2.imread('assets/eye.png')

vid = cv2.VideoCapture(0)
vid.set(3,1280)
vid.set(4,960)
time1 = 0
while True:
    ret, frame = vid.read()
    if not ret:
        continue

    _, face_mesh_results = detectFacialLandmarks(frame, mode='image', display=False)
    if face_mesh_results.multi_face_landmarks:
        # output_image, _ = isOpen(frame, face_mesh_results, 'MOUTH', threshold=15, display=False)

        # Get the left eye isOpen status of the person in the frame.
        _, left_eye_status = isOpen(frame, face_mesh_results, 'LEFT EYE', 
                                        threshold=4.5 , display=False)
        
        # Get the right eye isOpen status of the person in the frame.
        _, right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                         threshold=4.5, display=False)

        # Iterate over the found faces.
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            
            # Check if the left eye of the face is open.
            if left_eye_status[face_num] == 'OPEN':
                
                # Overlay the left eye image on the frame at the appropriate location.
                frame = overlay(frame, eye, face_landmarks,
                                'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
            
            # Check if the right eye of the face is open.
            if right_eye_status[face_num] == 'OPEN':
                
                # Overlay the right eye image on the frame at the appropriate location.
                frame = overlay(frame, eye, face_landmarks,
                                'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
            
            # # Check if the mouth of the face is open.
            # if mouth_status[face_num] == 'OPEN':
                
            #     # Overlay the smoke animation on the frame at the appropriate location.
            #     frame = overlay(frame, smoke_frame, face_landmarks, 
            #                     'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)

    time2 = time()
    if (time2 - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()