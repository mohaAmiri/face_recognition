import pandas as pd
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import redis

r = redis.Redis(host='localhost', port=6379, db=0, charset="utf-8")

""" Create Face App """
face_app = FaceAnalysis(name='buffalo_sc', root="insightface_models", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


def register():
    """ Register """
    # Get name and role
    while True:
        person_name = input('Enter your name: ')
        role = input('Please choose the role:1. Employee, 2. Manager:')
        if role in ('1', '2'):
            if role == '1':
                person_role = 'Employee'
            else:
                person_role = 'Manager'
            break
        else:
            print('Invalid input, try again')

    key = person_name + '@' + person_role

    print("Collecting face information ...")

    # Collect samples
    face_embeddings = []
    sample = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret is False:
            print('Unable to read camera')
            break
        # get results from insight face
        results = face_app.get(frame, max_num=1)
        for res in results:
            sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # facial features
            embeddings = res['embedding']
            face_embeddings.append(embeddings)
        if sample >= 200:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    data = np.asarray(face_embeddings).mean(axis=0)
    # convert to bytes
    data = data.tobytes()
    r.hset('new', key=key, value=data)
    print('Successfully registered!!!')


def ml_search_algorithm(data_frame, feature_column, test_vector, name_role=['Name', 'Role'], thresh=0.5):
    """ Cosine similarity search algorithm """
    data_frame = data_frame.copy()
    # This is saved data
    x_list = data_frame[feature_column].tolist()
    x = np.asarray(x_list)

    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    data_frame['cosine'] = similar_arr

    data_filter = data_frame.query(f'cosine >={thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        arg_max = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[arg_max][name_role]

    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


def face_detection(test_image, data_frame, feature_column, name_role=['Name', 'Role'], thresh=0.5):
    results = face_app.get(test_image)
    test_image_copy = test_image.copy()
    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(data_frame, feature_column, embeddings,
                                                       name_role, thresh)

        if person_name == 'Unknown':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(test_image_copy, (x1, y1), (x2, y2), color, 1)
        cv2.putText(test_image_copy, person_name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    return test_image_copy


def detection(retrieve_df):
    """ Face detection """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        pred_frame = face_detection(frame, retrieve_df, 'Facial_Features', ['Name', 'Role'], thresh=0.5)
        cv2.imshow('Press "ESC" to quit', pred_frame)

        if cv2.waitKey(1) == 27:  # esc
            break

    cap.release()
    cv2.destroyAllWindows()


def extract_data():
    print("Processing ...")
    try:
        retrieve_dict = r.hgetall('new')
        retrieve_series = pd.Series(retrieve_dict)
        retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
        index = retrieve_series.index
        index = list(map(lambda x: x.decode(), index))
        retrieve_series.index = index

        retrieve_df = retrieve_series.to_frame().reset_index()
        retrieve_df.columns = ['name_role', 'Facial_Features']

        retrieve_df[['Name', 'Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
        return retrieve_df
    except ValueError:
        print("*** Should Register at least one person!!! ")
        return None


while True:
    print('=' * 30)
    print("1: Register")
    print("2: Recognition")
    print("q: Quite")
    user_choice = input("Please select an option:\n")
    print('=' * 30)

    if user_choice == '1':
        register()
    if user_choice == '2':
        data_frame = extract_data()
        if data_frame is not None:
            detection(data_frame)
    elif user_choice == 'q':
        break
