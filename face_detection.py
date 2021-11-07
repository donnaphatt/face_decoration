import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_detection

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
# Initialize the mediapipe face mesh class.
mp_face_mesh = mp.solutions.face_mesh

# Setup the face landmarks function for images.
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)

# Setup the face landmarks function for videos.
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                         min_detection_confidence=0.5,min_tracking_confidence=0.3)

def face_detect_box(img):
    face_detection_results = face_detection.process(img[:,:,::-1])
    return face_detection_results

def print_face_info(face_info):
    if face_info.detections:
    # Iterate over the found faces.
        for face_no, face in enumerate(face_info.detections):
            
            # Display the face number upon which we are iterating upon.
            print(f'FACE NUMBER: {face_no+1}')
            print('---------------------------------')
            
            # Display the face confidence.
            print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
            
            # Get the face bounding box and face key points coordinates.
            face_data = face.location_data
            
            # Display the face bounding box coordinates.
            print(f'\nFACE BOUNDING BOX:\n{face_data.relative_bounding_box}')
            
            # Iterate two times as we only want to display first two key points of each detected face.
            for i in range(2):
                # Display the found normalized key points.
                print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
                print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}') 

def draw_box_position(img, face_info):
    # Create a copy of the sample image to draw the bounding box and key points.
    img_copy = img[:,:,::-1].copy()
    
    # Check if the face(s) in the image are found.
    if face_info.detections:
        
        # Iterate over the found faces.
        for face_no, face in enumerate(face_info.detections):
            
            # Draw the face bounding box and key points on the copy of the sample image.
            mp_drawing.draw_detection(image=img_copy, detection=face, 
                                    keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                thickness=2,
                                                                                circle_radius=2))
    # # Specify a size of the figure.
    # fig = plt.figure(figsize = [10, 10])
    
    # # Display the resultant image with the bounding box and key points drawn, 
    # # also convert BGR to RGB for display. 
    # plt.title("Resultant Image");plt.axis('off');plt.imshow(img_copy);plt.show()
    return img_copy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("kid.jpeg")
    face_info = face_detect_box(img)
    print_face_info(face_info)
    result_img = draw_box_position(img, face_info)
    plt.imshow(result_img)
    plt.show()