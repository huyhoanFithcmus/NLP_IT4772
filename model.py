import re
import numpy as np
import tensorflow as tf
from pyvi import ViTokenizer, ViPosTagger
from data_utils import get_sentence_indices,clean_sentences
from load_data import word2idx

# Mô hình phân tích cảm xúc của câu
class SentimentAnalysisModel(tf.keras.Model):
    """
    Parameter
    ----------
    word2vec: numpy.array
        word vectors
    lstm_layers: list
        list of lstm layers, lstm cuối cùng sẽ chỉ trả về output của lstm cuối cùng
    dropout_layers: list
        list of dropout layers
    dense_layer: Keras Dense Layer
        lớp dense layer cuối cùng nhận input từ lstm,
        đưa ra output bằng số lượng class thông qua hàm softmax
    """

    def __init__(self, word2vec, lstm_units, n_layers, num_classes, dropout_rate=0.25):
        """
        Khởi tạo mô hình

        Paramters
        ---------
        word2vec: numpy.array
            word vectors
        lstm_units: int
            số đơn vị lstm
        n_layers: int
            số layer lstm xếp chồng lên nhau
        num_classes: int
            số class đầu ra
        dropout_rate: float
            tỉ lệ dropout giữa các lớp
        """
        super().__init__(name='sentiment_analysis')

        # Khởi tạo các đặc tính của model
        self.word2vec = word2vec

        self.lstm_layers = []  # List chứa các tầng LSTM
        self.dropout_layers = []  # List chứa các tầng dropout

        # Khởi tạo các layer
        for i in range(n_layers):
            new_lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)
            # return_sequences = True trả ra chuỗi có cùng độ dài
            #                  = False trả ra một vector
            self.lstm_layers.append(new_lstm)
            new_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
            self.dropout_layers.append(new_dropout)

        # Tầng cuối cùng
        new_lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=False)
        self.lstm_layers.append(new_lstm)

        self.dense_layer = tf.keras.layers.Dense(num_classes, activation="softmax")


    def call(self, inputs):
        # Thực hiện các bước biến đổi khi truyền thuận input qua mạng
        inputs = tf.cast(inputs, tf.int32)
        #print(inputs)
        # Input hiện là indices, cần chuyển sang dạng vector

        x = tf.nn.embedding_lookup(self.word2vec, inputs)
        #print(x)
        # Truyền thuận inputs lần lượt qua các tầng
        # ở mỗi tầng, truyền input qua các layer: lstm > dropout
        n_layers = len(self.dropout_layers)

        for i in range(n_layers):
            x = self.lstm_layers[i](x)
            x = self.dropout_layers[i](x)
            #print(x)

        x = self.lstm_layers[-1](x)
        #print(x)
        x = self.dense_layer(x)
        #print(x)
        # Gán giá trị tầng cuối cùng vào out và trả về
        out = x

        return out


def predict(sentence, model, _word_list, _max_seq_length, word2idx):
    """
    Dự đoán cảm xúc của một câu

    Parameters
    ----------
    sentence: str
        câu cần dự đoán
    model: model keras
        model keras đã được train/ load trọng số vừa train
    _word_list: numpy.array
        danh sách các từ đã biết
    _max_seq_length: int
        giới hạn số từ tối đa trong mỗi câu

    Returns
    -------
    int
        0 nếu là negative, 1 nếu là positive
    """
    # Xóa các kí tự kéo dài aaaaa,bbbbbbb
    sentence = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), sentence, flags=re.IGNORECASE)
    # Tokenize/Tách từ trong câu
    # Sử dụng hàm word_tokenize vừa import ở trên
    tokenized_sent = ViTokenizer.tokenize(sentence)

    # Đưa câu đã tokenize về dạng input_data thích hợp để truyền vào model
    ### START CODE HERE
    tokenized_sent = clean_sentences(tokenized_sent)
    tokenized_sent = ViTokenizer.tokenize(tokenized_sent)
    input_data = get_sentence_indices(tokenized_sent, _max_seq_length, _word_list)
    input_data = input_data.reshape(-1, _max_seq_length)

    # Truyền input_data qua model để nhận về xác suất các nhãn
    # Chọn nhãn có xác suất cao nhất và return
    pred = model(input_data)
    predictions = tf.argmax(pred, 1).numpy().astype(np.int32)
    prob_pos = pred.numpy()[0][1]

    return predictions, prob_pos
