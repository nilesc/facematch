import io
import sys
import sqlite3
import numpy as np
from face_embed import Embedder
from pose_estimator import PoseEstimator


def process_video(video, embedder, pose_estimator):
    embedding = embedder.embed(np.expand_dims(video[0], 0))
    poses = pose_estimator.estimate_pose(video)
    return (embedding, [(video[i], poses[i])\
            for i in range(len(video))])

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


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Requires an embedder weights file and a pose weights file')

    facenet_protobuf = sys.argv[1]
    pose_weights = sys.argv[2]

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

    batch_size = 10
    # Replace when we have actual data
    dummy_images = np.random.rand(27, 200, 200, 3)
    batched_images = [dummy_images[i:i+batch_size] for i in range(0, len(dummy_images), batch_size)]
    for batch in batched_images:
        processed = process_video(batch, embedder, pose_estimator)
        print(processed)

    c.execute('INSERT INTO videos (embedding) values (?)', (np.random.rand(128),))
    c.execute('INSERT INTO videos (embedding) values (?)', (np.random.rand(128),))
    c.execute('INSERT INTO frames (video_id) values (?)', (1,))
    c.execute('INSERT INTO frames (video_id) values (?)', (1,))
    c.execute('SELECT video_id FROM frames')
    data = c.fetchall()
    print(data)
