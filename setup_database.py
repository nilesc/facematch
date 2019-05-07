import io
import os
import csv
import sys
import math
import sqlite3
import numpy as np
from PIL import Image
from face_embed import Embedder
from pose_estimator import PoseEstimator
from helpers import crop_to_face, get_normalized_landmarks
from progress.bar import IncrementalBar


# Based on: https://www.pythonforthelab.com/blog/storing-data-with-sqlite/
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def populate_database(video_directory,
                      embedder,
                      pose_estimator,
                      cursor,
                      num_people=None):
    folders = [f for f in os.listdir(video_directory)
               if os.path.isdir(os.path.join(video_directory, f))]
    for person_number, folder in enumerate(folders):
        if person_number == num_people:
            return

        frame_info = os.path.join(video_directory,
                                  folder + '.labeled_faces.txt')

        with open(frame_info) as info_file:
            csv_data = csv.reader(info_file, delimiter=',')
            embedding = None
            csv_data = list(csv_data)
            bar = IncrementalBar(f'Adding person {person_number} of {num_people}', max=len(csv_data))
            for frame_number, row in enumerate(csv_data):
                image_path = row[0].replace('\\', '/')
                image_path = os.path.join(video_directory, image_path)
                image = Image.open(image_path)
                image = crop_to_face(image)

                if embedding is None:
                    embedding = embedder.embed(image)
                    embedding = embedding.flatten()
                    c.execute('INSERT INTO videos (id, embedding) values' +
                              '  (?, ?)',
                              (person_number, embedding))

                pose = pose_estimator.estimate_pose(image)
                landmarks = get_normalized_landmarks(image)

                c.execute('INSERT INTO frames (video_id, image_path, pose, landmarks)' +
                          ' values (?, ?, ?, ?)',
                          (frame_number, image_path, pose, landmarks))
                bar.next()
        print()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Requires a data directory, an embedder weights file,' +
                 'and a pose estimator weights file')

    video_directory = sys.argv[1]
    facenet_protobuf = sys.argv[2]
    pose_weights = sys.argv[3]

    embedder = Embedder(facenet_protobuf)
    pose_estimator = PoseEstimator(pose_weights)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    database_file = 'video_database.db'
    conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES)

    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS videos
            (id INTEGER PRIMARY KEY,
             embedding array)''')
    c.execute('''CREATE TABLE IF NOT EXISTS frames
            (video_id INTEGER,
             image_path STRING,
             pose array,
             landmarks array,
             FOREIGN KEY(video_id) REFERENCES videos(id))''')

    # changed from 3 to 2
    populate_database(video_directory, embedder, pose_estimator, c, 2)
    batch_size = 10

    conn.commit()
