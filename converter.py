import tfcoreml as tf_converter


tf_converter.convert(tf_model_path='inception_v1_2016_08_28_frozen.pb',
                     mlmodel_path='InceptionV1.mlmodel',
                     output_feature_names=['InceptionV1/Logits/Predictions/Softmax:0'],
                     image_input_names='input:0',
                     class_labels='imagenet_slim_labels.txt')
