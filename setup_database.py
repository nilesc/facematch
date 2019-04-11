import io
import os
import csv
import sys
import sqlite3
import numpy as np
from PIL import Image
from face_embed import Embedder
from pose_estimator import PoseEstimator


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
            for frame_number, row in enumerate(csv_data):
                image_path = row[0].replace('\\', '/')
                face_center_x = int(row[2])
                face_center_y = int(row[3])
                bounding_box_dimensions_x = int(row[4])
                bounding_box_dimensions_y = int(row[5])
                image_path = os.path.join(video_directory, image_path)
                image = Image.open(image_path)
                upper_left_corner_x = (face_center_x -
                                       bounding_box_dimensions_x/2)
                upper_left_corner_y = (face_center_y -
                                       bounding_box_dimensions_y/2)
                image.crop((upper_left_corner_x,
                            upper_left_corner_y,
                            bounding_box_dimensions_x,
                            bounding_box_dimensions_y))

                cropped = np.array(image)
                pose_image = np.expand_dims(cropped, 0)
                if embedding is None:
                    image_dimension = 160
                    embedding_image = image.resize((image_dimension,
                                                      image_dimension))
                    embedding_image = np.array(embedding_image)
                    embedding_image = np.expand_dims(embedding_image, 0)
                    embedding = embedder.embed(embedding_image)
                    embedding = embedding.flatten()
                    c.execute('INSERT INTO videos (id, embedding) values' +
                              '  (?, ?)',
                              (person_number, embedding))

                pose = pose_estimator.estimate_pose(pose_image)
                c.execute('INSERT INTO frames (video_id, image_path, pose)' +
                          ' values (?, ?, ?)',
                          (frame_number, image_path, pose))
                print(person_number)


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
             FOREIGN KEY(video_id) REFERENCES videos(id))''')

    populate_database(video_directory, embedder, pose_estimator, c, 3)
    batch_size = 10

    c.execute('SELECT * FROM frames')
    data = c.fetchall()
    conn.commit()
    print(data)
