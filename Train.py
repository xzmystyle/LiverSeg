import argparse
from keras.layers import Dropout,Flatten,Dense,GlobalAvgPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras import applications, optimizers
from keras.models import Sequential, Model

import matplotlib.pyplot as plt


def set_args():
    parser = argparse.ArgumentParser(description='Liver Patches classification parameters')
    parser.add_argument("--img_channel", type=int, default=3)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--train_samples", type=int, default=3000, help='every epoch run train_sample images')
    parser.add_argument("--valid_samples", type=int, default=1000,help='every epoch run valid_sample images')
    parser.add_argument("--epochs", type=int, default=20, help='num of epochs for training')
    parser.add_argument("--batch_size",type=int, default=20, help='num of imgs training for a epoch')
    parser.add_argument("--lr", type=float, default=1.0e-5, help='learning rate')
    parser.add_argument("--layer_strategy", type=str, default='flatten', help='a strategy for output layer')
    parser.add_argument("--model_weight", type=str, default='imagenet', help='weight model used')
    parser.add_argument('--model_name', type=str, default='VGG16', help='transfer learning based on model')
    parser.add_argument("--train_dir", type=str, default='./data/train_data', help='training data directory')
    parser.add_argument("--valid_dir", type=str,default='./data/valid_data', help='validation data directory')
    parser.add_argument("--save_path", type=str, default='./model', help='model save path')
    parser.add_argument("--plot", action='store_true', default=True)

    args = parser.parse_args()
    return args


def set_final_layer_flatten(model):
    x = model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model_flatten = Model(inputs=model.input, outputs=predictions)

    return model_flatten


def set_final_layer_gap(model):
    x = model.output
    x = GlobalAvgPool2D((8,8))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model_gap = Model(inputs=model.input, outputs=predictions)

    return model_gap


def set_model(args):
    if args.model_name == 'VGG16':
        model = applications.VGG16(weights=args.model_weight, include_top=False,
                                   input_shape=(args.img_width, args.img_height, args.img_channel))
        print('model loaded')
        # model.summary()

        for layer in model.layers:
            layer.trainable = False

        if args.layer_strategy == 'flatten':
            model_final = set_final_layer_flatten(model)
            return model_final
        elif args.layer_strategy == 'gap':
            model_final = set_final_layer_gap(model)
            return model_final

    elif args.model_name == 'VGG19':
        model = applications.VGG19(weights=args.model_weight, include_top=False,
                                   input_shape=(args.img_width, args.img_height, args.img_channel))
        print('model loaded')

        for layer in model.layers:
            layer.trainable = False

        if args.layer_strategy == 'flatten':
            model_final = set_final_layer_flatten(model)
            return model_final
        elif args.layer_strategy == 'gap':
            model_final = set_final_layer_gap(model)
            return model_final

    elif args.model_name == 'ResNet50':
        model = applications.resnet50.ResNet50(weights=args.model_weight, include_top=False,
                                   input_shape=(args.img_width, args.img_height, args.img_channel))
        print('model loaded')

        for layer in model.layers:
            layer.trainable = False

        if args.layer_strategy == 'flatten':
            model_final = set_final_layer_flatten(model)
            return model_final
        elif args.layer_strategy == 'gap':
            model_final = set_final_layer_gap(model)
            return model_final

    elif args.model_name == 'InceptionV3':
        model = applications.InceptionV3(weights = args.model_weight, include_top=False,
                                         input_shape=(args.img_width, args.img_height, args.img_channel))
        print('model loaded')
        # model.summary()

        for layer in model.layers:
            layer.trainable = False

        if args.layer_strategy == 'flatten':
            model_final = set_final_layer_flatten(model)
            return model_final
        elif args.layer_strategy == 'gap':
            model_final = set_final_layer_gap(model)
            return model_final

    else:
        raise AssertionError('unknown model')


def train(args):
    train_gen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   rescale=1./255, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

    valid_gen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   rescale=1./255, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

    train_generator = train_gen.flow_from_directory(args.train_dir, target_size=(args.img_width, args.img_height),
                                                    batch_size=args.batch_size, class_mode='categorical')

    valid_generator = valid_gen.flow_from_directory(args.train_dir, target_size=(args.img_width, args.img_height),
                                                    batch_size=args.batch_size, class_mode='categorical')

    model = set_model(args)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=args.lr, momentum=0.9),
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator,
                                  epochs=args.epochs,
                                  steps_per_epoch=int(args.train_samples//args.batch_size),
                                  validation_data=valid_generator,
                                  validation_steps=int(args.valid_samples//args.batch_size))

    model.save(args.save_path)
    if args.plot:
        train_plot(history)


def train_plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('loss.png')


if __name__ == '__main__':
    args = set_args()
    train(args)


