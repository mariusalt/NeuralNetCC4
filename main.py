# mlp for multiclass classification
from numpy import argmax
import pandas as pd
import csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import kerastuner as kt
import tensorflow as tf
from tensorflow.keras import regularizers
import itertools
from datetime import datetime
from sklearn.utils import class_weight
import numpy as np
from keras.layers import GaussianNoise


start_date= datetime.now()
start_date=start_date.strftime("%d/%m/%Y %H:%M:%S")


# load the dataset
path = '/home/mat/Desktop/cond_coop/data/sesca3_uc.csv'
df = read_csv(path)
#print(df1.head(3))
# split into input and output columns
#df = df1.drop(df1.columns[0:19], axis=1)
print(df.head(3))
df = df.drop('choice1', 1)
df = df.drop('cooptyp.1', 1)
print(list(df))
X = df.values[:, 1:]
y = df.values[:, 0]

# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = {i : class_weights[i] for i in range(3)}

# define model

def model_builder(hp):
    layers=lay[-1]
    print(layers)
    model = Sequential()

    li_unit = ['units' + str(p+1) for p in range(layers)]
    li_unit_name = ['hp_units' + str(p+1) for p in range(layers)]
    li_unit_value = [hp.Int(li_unit[p], min_value=10, max_value=150, step=1)  for p in range(layers)]
    unit_dict = dict(zip(li_unit_name, li_unit_value))
    print(unit_dict)
    locals().update(dict(itertools.islice(unit_dict.items(), layers)))

    li_drop = ['dropout' + str(p+1) for p in range(layers)]
    li_drop_name = ['hp_dropout' + str(p+1) for p in range(layers)]
    li_drop_value = [hp.Float(li_drop[p], 0, 0.5, step=0.1, default=0.5) for p in range(layers)]
    drop_dict = dict(zip(li_drop_name, li_drop_value))
    print(drop_dict)
    locals().update(dict(itertools.islice(drop_dict.items(), layers)))
    lreg1=hp.Choice('lreg1',[0.01,0.001,0.0001])
    lreg2 = hp.Choice('lreg2', [0.01, 0.001, 0.0001])
    noise = hp.Choice('noise', [0.00, 0.025, 0.05,0.1,0.2])

    model.add(GaussianNoise(noise))
    for z in range(layers):
        # add dense layer

        model.add(Dense(units=unit_dict[li_unit_name[z]], activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l1_l2(l1=lreg1,l2=lreg2)))
        model.add(tf.keras.layers.Dropout(drop_dict[li_drop_name[z-1]]))

    model.add(Dense(6, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3,1e-4,1e-5])

    model.compile(optimizer=keras.optimizers.SGD(hp_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
out=[]
lay=[p for p in range(1,4)]*2
lay=sorted(lay)
print(lay)
for i in range(6):
    name='round0505_'+ str(i)
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=100,
                         factor=3,
                         hyperband_iterations=2,
                         directory='my_dir',
                         project_name=name)

   # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)


    tuner.search(X_train, y_train, epochs=200, batch_size=81, validation_split=0.3,class_weight=class_weights)#, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    #print(best_hps.get(.))
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]
   # es = EarlyStopping(monitor='val_accuracy', patience=20)
    history = best_model.fit(X_train, y_train, epochs=200, batch_size=81, verbose=0, validation_split=0.3,
                             class_weight=class_weights)#callbacks=[es],
    score1 = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss:{score1[0]} / Test accuracy: {score1[1]}')
    history = best_model.fit(X_train, y_train, epochs=200, batch_size=81, verbose=0, validation_split=0.3,
                             class_weight=class_weights)#callbacks=[es],
    score2 = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss:{score2[0]} / Test accuracy: {score2[1]}')
    history = best_model.fit(X_train, y_train, epochs=200, batch_size=81, verbose=0, validation_split=0.3,
                             class_weight=class_weights)#callbacks=[es],
    score3 = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss:{score3[0]} / Test accuracy: {score3[1]}')

    if lay[-1]==4:
        out.append([lay.pop(), best_hps.get('learning_rate'),best_hps.get('lreg1'),best_hps.get('lreg2'),best_hps.get('noise'), best_hps.get('units1'), best_hps.get('units2'),
                    best_hps.get('units3'), best_hps.get('units4'),best_hps.get('dropout1'),
                    best_hps.get('dropout2'), best_hps.get('dropout3'), best_hps.get('dropout4'),
                    score1[0], score1[1], score2[0], score2[1],
                    score3[0], score3[1]])
    elif lay[-1]==3:
        out.append([lay.pop(), best_hps.get('learning_rate'),best_hps.get('lreg1'),best_hps.get('lreg2'),best_hps.get('noise'), best_hps.get('units1'), best_hps.get('units2'),
                    best_hps.get('units3'),'none',best_hps.get('dropout1'),
                    best_hps.get('dropout2'), best_hps.get('dropout3'),'none',  score1[0], score1[1], score2[0], score2[1],
                    score3[0], score3[1]])
    elif lay[-1]==2:
        out.append([lay.pop(), best_hps.get('learning_rate'),best_hps.get('lreg1'),best_hps.get('lreg2'),best_hps.get('noise'), best_hps.get('units1'), best_hps.get('units2'),'none','none',
                    best_hps.get('dropout1'),
                    best_hps.get('dropout2'),'none','none',  score1[0], score1[1], score2[0], score2[1],
                    score3[0], score3[1]])
    elif lay[-1]==1:
        out.append([lay.pop(), best_hps.get('learning_rate'),best_hps.get('lreg1'),best_hps.get('lreg2'),best_hps.get('noise'), best_hps.get('units1'),'none','none',best_hps.get('dropout1'),'none','none','none',
                      score1[0], score1[1], score2[0], score2[1],
                    score3[0], score3[1]])

print(out)


with open("/home/mat/Desktop/cond_coop/trials/sesca2/20210504main3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out)

end_date= datetime.now()
end_date=end_date.strftime("%d/%m/%Y %H:%M:%S")
print(start_date)
print(end_date)
print("session started" ,start_date, "and ended", end_date)

#,layer2:{best_hps.get('units2')}
#,layer4:{best_hps.get('units4')}
#,layer3:{best_hps.get('units3')}
#, dropout1:{best_hps.get('dropout1')}, dropout2:{best_hps.get('dropout2')}.


#es = EarlyStopping(monitor='val_acc', patience=5)

# fit the model
#history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.3, callbacks=[es])
# evaluate the model
#loss, acc = model.evaluate(X_test, y_test, verbose=0)
#print('Test Accuracy: %.3f' % acc)
# make a prediction
#row = [5.1,3.5,1.4,0.2]
#yhat = model.predict([row])
#print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

#print(history.history.keys())
#pyplot.title('Learning Curves')
#pyplot.xlabel('Epoch')
#pyplot.ylabel('Cross Entropy')
#pyplot.plot(history.history['acc'], label='train')
#pyplot.plot(history.history['val_acc'], label='val')
#pyplot.legend()
#pyplot.show()








 #   print(f"""
 #   learning rate:{best_hps.get('learning_rate')},layer1:{best_hps.get('units1')},layer2:{best_hps.get('units2')},layer2:{best_hps.get('units3')},layer2:{best_hps.get('units4')},layer2:{best_hps.get('units5')}, dropout1:{best_hps.get('dropout1')}, dropout2:{best_hps.get('dropout2')}, dropout3:{best_hps.get('dropout3')}, dropout4:{best_hps.get('dropout4')}.
 #   """)

 #   out.append([best_hps.get('learning_rate'),best_hps.get('units1'), best_hps.get('units2'), best_hps.get('units3'),best_hps.get('units4'),best_hps.get('units5'),best_hps.get('dropout1'),best_hps.get('dropout2'),best_hps.get('dropout3'),best_hps.get('dropout4'),best_hps.get('dropout5')])


#print(out)


#with open("/home/mat/Desktop/perf1.csv", "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow([ 'lr','layer1','layer2','layer3','layer4','layer5','dropout1','dropout2','dropout3','dropout4','dropout5'])
#    writer.writerows(out)
#,layer2:{best_hps.get('units2')}
#,layer4:{best_hps.get('units4')}
#,layer3:{best_hps.get('units3')}
#, dropout1:{best_hps.get('dropout1')}, dropout2:{best_hps.get('dropout2')}.


#es = EarlyStopping(monitor='val_acc', patience=5)

# fit the model
#history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.3, callbacks=[es])
# evaluate the model
#loss, acc = model.evaluate(X_test, y_test, verbose=0)
#print('Test Accuracy: %.3f' % acc)
# make a prediction
#row = [5.1,3.5,1.4,0.2]
#yhat = model.predict([row])
#print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

#print(history.history.keys())
#pyplot.title('Learning Curves')
#pyplot.xlabel('Epoch')
#pyplot.ylabel('Cross Entropy')
#pyplot.plot(history.history['acc'], label='train')
#pyplot.plot(history.history['val_acc'], label='val')
#pyplot.legend()
#pyplot.show()



