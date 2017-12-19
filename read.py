import tensorflow as tf
from synset import *
import skimage.io
import skimage.transform
import numpy as np
import glob
import json


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
    saver = tf.train.import_meta_graph('ResNet-L50.meta')
    saver.restore(session, 'ResNet-L50.ckpt')

def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1)
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print("Top5: ", top5)
    return top1

def main():
    with tf.Session() as sess:
        restore_session(sess)

        graph = tf.get_default_graph()

        resnet_hash = graph.get_tensor_by_name("fc/xw_plus_b:0")

        prob_tensor = graph.get_tensor_by_name("prob:0")
        for node in prob_tensor.op.inputs:
            print('was', node, node.op)
        images = graph.get_tensor_by_name("images:0")
        for op in graph.get_operations():
            # print('->', op.name, '<-', op.name == 'prob')
            if op.name == 'prob':
                print('wow', op.inputs)

        print("graph restored")

        filenames = glob.glob('data/*.jpg')

        resized_images = []
        for filename in filenames:
            resized_images.append(load_image(filename))

        real_batch = np.stack(resized_images)
        print('real batch', real_batch.shape)

        # batch = img.reshape((1, 224, 224, 3))
        # print(batch.shape)

        ranks, hashes = sess.run([prob_tensor, resnet_hash], feed_dict={images: real_batch})
        all_hashes = {}
        for filename, prob, img_hash in zip(filenames, ranks, hashes):
            print(filename)
            print_prob(prob)
            all_hashes[filename] = img_hash.tolist()
        
        np.save('hashes.npy', hashes)
        with open('hashes.json', 'w') as f:
            f.write(json.dumps(all_hashes, indent=4))


if __name__ == '__main__':
    main()
