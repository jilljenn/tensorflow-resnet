import tensorflow as tf
from synset import *
import skimage.io
import skimage.transform
import numpy as np
import glob
import json
import os.path
from datetime import datetime


TRAIN_FOLDER = 'data'
BATCH_SIZE = 40


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

def restore_session(session):
    saver = tf.train.import_meta_graph('tensorflow-resnet-pretrained-20160509/ResNet-L152.meta')
    saver.restore(session, 'tensorflow-resnet-pretrained-20160509/ResNet-L152.ckpt')

def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    # print("Top1: ", top1)
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    # print("Top5: ", top5)
    return top5

def get_batches(l):
    n = len(l)
    for i in range(0, n, BATCH_SIZE):
        yield l[i:i + BATCH_SIZE]

def main():
    # Load existing DB
    try:
        with open('filenames.txt') as f:
            db_filenames = f.read().splitlines()
        hashes = [np.load('hashes.npy')]
    except:
        db_filenames = []
        hashes = []

    with tf.Session() as sess:
        restore_session(sess)

        graph = tf.get_default_graph()
        resnet_hash = graph.get_tensor_by_name("fc/xw_plus_b:0")  # ResNet hashes
        prob_tensor = graph.get_tensor_by_name("prob:0")  # Prob on classes
        images = graph.get_tensor_by_name("images:0")
        print("Graph restored")

        filenames = [filename for filename in glob.glob(os.path.join(TRAIN_FOLDER, '*.jpg')) if filename not in db_filenames]
        print('Compute for', len(filenames), '->', filenames)

        topfives = []
        for batch in get_batches(filenames):
            start = datetime.now()
            resized_images = []
            for filename in batch:
                resized_images.append(load_image(filename))

            real_batch = np.stack(resized_images)
            print('real batch', real_batch.shape)

            # batch = img.reshape((1, 224, 224, 3))
            # print(batch.shape)

            ranks, batch_hashes = sess.run([prob_tensor, resnet_hash], feed_dict={images: real_batch})
            hashes.append(batch_hashes)
            # all_hashes = {}
            
            for filename, prob, img_hash in zip(batch, ranks, batch_hashes):
                # print(filename)
                topfives.append(str(print_prob(prob)))
                # all_hashes[filename] = img_hash.tolist()
            print('batch', datetime.now() - start)
        all_hashes = np.vstack(hashes)
        print('all hashes', all_hashes.shape)

        with open('filenames.txt', 'a+') as f:
            f.write('\n'.join(filenames))
        with open('topfives.txt', 'a+') as f:
            f.write('\n'.join(topfives))

        np.save('hashes.npy', all_hashes.astype(np.float64))


if __name__ == '__main__':
    main()
