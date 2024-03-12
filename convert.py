import tensorflow as tf

def convert_checkpoint_to_h5(checkpoint_prefix, output_h5_path):
    # Tạo mô hình mẫu
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Tạo optimizer và compile mô hình (chú ý rằng bạn cần tạo mô hình có cùng kiến trúc)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Tạo một đối tượng checkpoint từ đường dẫn checkpoint
    checkpoint = tf.train.load_checkpoint(checkpoint_prefix)

    # Lấy danh sách các biến trong checkpoint
    var_list = checkpoint.get_variable_to_shape_map()

    # Lấy các lớp trong mô hình Keras
    model_layers = [layer for layer in model.layers if layer.trainable]

    # Gán trọng số từ checkpoint vào các lớp tương ứng của mô hình Keras
    for layer in model_layers:
        layer_name = layer.name
        if layer_name in var_list:
            layer.set_weights([checkpoint.get_tensor(layer_name + '/kernel:0'),
                               checkpoint.get_tensor(layer_name + '/bias:0')])

    # Lưu trọng số dưới dạng .h5
    model.save_weights(output_h5_path)

    print("Conversion completed successfully.")

# Đường dẫn đến file checkpoint (thay đổi tùy thuộc vào đường dẫn của bạn)
checkpoint_prefix = '/workspaces/NLP_IT4772/modelver7/ckpt_0.8614501076812635'

# Đường dẫn để lưu trọng số dưới dạng .h5
output_h5_path = '/workspaces/NLP_IT4772/modelver7/ckpt_0.8614501076812635.weights.h5'

# Gọi hàm để chuyển đổi
convert_checkpoint_to_h5(checkpoint_prefix, output_h5_path)
