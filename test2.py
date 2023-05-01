import jieba
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
import json
from keras.models import load_model

# 准备数据集
tweets = []
for line in open('./example.json', 'r'):
    tweets.append(json.loads(line))

articles = []
summaries = []
for item in tweets:
    if 'DRECONTENT' in item['_source'] and 'DRETITLE' in item['_source']:
        articles = articles + [item['_source']['DRECONTENT']]
        summaries = summaries + [item['_source']['DRETITLE']]

# 分词
articles = [' '.join(jieba.cut(article)) for article in articles]
summaries = [' '.join(jieba.cut(summary)) for summary in summaries]

# 特征提取
tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(articles + summaries)
word_index = tokenizer.word_index
index_word = tokenizer.index_word
input_vocab_size = len(word_index) + 1
output_vocab_size = input_vocab_size
max_article_len = max([len(article.split(' ')) for article in articles])
max_summary_len = max([len(summary.split(' ')) for summary in summaries])

encoder_inputs = []
for article in articles:
    seq = tokenizer.texts_to_sequences([article])[0]
    encoder_inputs.append(seq)
encoder_inputs = pad_sequences(encoder_inputs, maxlen=max_article_len, padding='post', truncating='post')

decoder_inputs = []
decoder_outputs = []
for summary in summaries:
    seq = [word_index['<start>']] + tokenizer.texts_to_sequences([summary])[0]
    decoder_inputs.append(seq[:-1])
    decoder_outputs.append(seq[1:])
decoder_inputs = pad_sequences(decoder_inputs, maxlen=max_summary_len, padding='post')
decoder_outputs = pad_sequences(decoder_outputs, maxlen=max_summary_len, padding='post')

# 将decoder_inputs进行one-hot编码
decoder_onehot_inputs = to_categorical(decoder_inputs, num_classes=output_vocab_size)

# 定义模型参数
latent_dim = 256

# 定义编码器模型
encoder_inputs_placeholder = Input(shape=(max_article_len,))
enc_emb = Embedding(input_vocab_size, latent_dim)(encoder_inputs_placeholder)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# 定义解码器模型
decoder_inputs_placeholder = Input(shape=(max_summary_len,))
dec_emb_layer = Embedding(output_vocab_size, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs_placeholder)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# 定义Attention层
attn_layer = Attention(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# 合并Attention层的输出和解码器LSTM的输出
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

# 全连接层输出
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

# 定义模型
model = Model(inputs=[encoder_inputs_placeholder, decoder_inputs_placeholder], outputs=decoder_outputs)

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# 打印模型概述
model.summary()

# 划分训练集和验证集
X_train_encoder, X_val_encoder, X_train_decoder, X_val_decoder, y_train, y_val = train_test_split(encoder_inputs, decoder_inputs, decoder_onehot_inputs, test_size=0.2)

# 定义学习率衰减函数
def lr_scheduler(epoch):
    # 函数功能：定义学习率衰减函数，每10个epoch将学习率减小为原来的0.9倍
    lr = 0.001
    if epoch > 0 and epoch % 10 == 0:
        lr *= 0.9
    return lr

# 定义EarlyStopping和LearningRateScheduler回调函数
earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# 训练模型
history = model.fit([X_train_encoder, X_train_decoder], y_train,
                    validation_data=([X_val_encoder, X_val_decoder], y_val),
                    batch_size=64, epochs=50, verbose=1,
                    callbacks=[earlystop, lr_scheduler_callback])

model = load_model('my_model.h5')