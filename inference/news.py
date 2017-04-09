# Copyright 2017 Jin Fagang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================
import tensorflow as tf
from models.seq2seq import Seq2SeqModel
from dataset.sougou_news import SougouNews
import logging
import numpy as np


def run_training():

    news = SougouNews()
    batches = news.batch_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10,
                                      batch_size=100)
    print(batches)
    print(np.array(batches).shape)
    loss_track = []

    model = Seq2SeqModel()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            for batch in range(5000):
                batch_data = next(batches)
                feed_data = model.make_train_inputs(batch_data, batch_data)
                _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_data)
                loss_track.append(loss)

                if batch == 0 or batch % 500 == 0:
                    logging.info('batch {1}, mini batch loss: {2}'.format(batch, loss))
                    for i, (e_in, dt_pred) in enumerate(zip(
                            feed_data[model.encoder_inputs].T,
                            sess.run(model.decoder_prediction_train, feed_data).T
                    )):
                        print('  sample {}:'.format(i + 1))
                        print('    enc input           > {}'.format(e_in))
                        print('    dec train predicted > {}'.format(dt_pred))
        except KeyboardInterrupt:
            logging.info('training interrupted.')


def main():
    run_training()


if __name__ == '__main__':
    tf.app.run()
