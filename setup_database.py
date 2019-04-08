import io
import os
import csv
import sys
import sqlite3
import numpy as np
from face_embed import Embedder
from pose_estimator import PoseEstimator


def process_video(video, embedder, pose_estimator):
    embedding = embedder.embed(np.expand_dims(video[0], 0))
    poses = pose_estimator.estimate_pose(video)
    return (embedding, [(video[i], poses[i]) for i in range(len(video))])

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
                      num_videos=None):
    folders = [f for f in os.listdir(video_directory)
               if os.path.isdir(os.path.join(video_directory, f))]
    for number, folder in enumerate(folders):
        if not number < num_videos:
            return

        frame_info = os.path.join(video_directory,
                                  folder + '.labeled_faces.txt')

        with open(frame_info) as info_file:
            csv_data = csv.reader(info_file, delimiter=',')
            for row in csv_data:
                print(row)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Requires a data directory, an embedder weights file,' + \
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
    c.execute('DROP TABLE IF EXISTS videos')
    c.execute('DROP TABLE IF EXISTS frames')

    c.execute('''CREATE TABLE videos
            (id INTEGER PRIMARY KEY,
             embedding array)''')
    c.execute('''CREATE TABLE frames
            (video_id INTEGER,
             pose array,
             FOREIGN KEY(video_id) REFERENCES videos(id))''')

    populate_database(video_directory, embedder, pose_estimator, c, 50)
    batch_size = 10
    # Replace when we have actual data
    dummy_images = np.random.rand(27, 200, 200, 3)
    batched_images = [dummy_images[i:i+batch_size]
                      for i in range(0, len(dummy_images), batch_size)]

    for video_number, batch in enumerate(batched_images):
        embedding, frame_tuples = process_video(batch,
                                                embedder,
                                                pose_estimator)

        c.execute('INSERT INTO videos (id, embedding) values (?, ?)',
                  (video_number, np.random.rand(128),))
        for frame, pose in frame_tuples:
            c.execute('INSERT INTO frames (video_id, pose) values (?, ?)',
                      (video_number, pose))

    c.execute('SELECT pose FROM frames WHERE video_id=0')
    data = c.fetchall()
    print(data)
