from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

INPUT_SIZE = (299, 299)


class ModelBuilder:
    model_name = 'inceptionv3'

    def __init__(self, nb_classes, nb_hidden, lr, dropout):
        self.__nb_classes = nb_classes
        self.__nb_hidden = nb_hidden
        self.__lr = lr
        self.__dropout = dropout

    def get_model(self):

        # load pretrained inception v3 (automatic weight download) with no top
        base_model = InceptionV3(weights='imagenet', include_top=False)
        # get the output of the no top model
        x = base_model.output
        # add a global average pooling and a dense layer as in the inception v3 architecture
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.__nb_hidden, activation='relu', name='features')(x)
        if self.__dropout > 0:
            x = Dropout(self.__dropout)(x)
        # add dense with the number of class with softmax activation
        predictions = Dense(self.__nb_classes, activation='softmax')(x)

        # create the new model
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=SGD(lr=self.__lr, decay=1e-6, nesterov=True, momentum=0.9),
                      loss='categorical_crossentropy', metrics=['acc'])

        return model
