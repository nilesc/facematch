import sys
import sqlite3
import numpy as np
from setup_database import adapt_array, convert_array
from face_embed import Embedder
from pose_estimator import PoseEstimator


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Requires a database, an embedder weights file,' +
                 'and a pose estimator weights file')

    database_file = sys.argv[1]
    facenet_protobuf = sys.argv[2]
    pose_weights = sys.argv[3]

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES)

    embedder = Embedder(facenet_protobuf)
    pose_estimator = PoseEstimator(pose_weights)

    c = conn.cursor()
    c.execute('SELECT embedding FROM videos')
    embeddings = c.fetchall()
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
    input_embedding = np.random.rand(1, 128)
    input_repeated = np.repeat(input_embedding, embeddings.shape[0], 0)
    difference = embeddings - input_repeated
    norms = np.linalg.norm(difference, axis=1)
    closest_indices = np.argpartition(norms, 0)
    num_closest = 5
    candidates = closest_indices[-num_closest:]

    query = f"SELECT * FROM frames WHERE video_id IN {str(tuple(candidates))}"

    c.execute(query)
    frames = c.fetchall()
    print(frames)
