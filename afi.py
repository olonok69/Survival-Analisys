from tfdeepsurv.datasets import survival_df
from tfdeepsurv import dsnn
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from IPython import embed
pd.options.mode.chained_assignment = None
# from keras import backend as K
# from tensorflow.compat.v1.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



import logging
logger = logging.getLogger(__name__)


class afi_estimator(object):
    """
    # time per Berth CLass child of c_estimator , Implement specific methods which apply to the dataset to use
    # in this stimator
    """
    def __init__(self):
        """
        TODO
        """


        self.AFI_Asset_Class_Grouping_enc=None
        self.AFI_ELR_enc=None
        self.AFI_Engineering_Suffix_enc=None
        self.AFI_Grouping_Full_Name_enc=None
        self.AFI_HLC_enc=None
        self.AFI_System_Asset_Type_enc=None
        self.AFI_weather_station_enc=None
        self.EQUIP_CLASS_DESC_enc=None
        self.AFI_factors_norm=None
        self.input_data=None
        self.model=None
        self.watch_list=None
        self.pred=None
        self.pre_predictions=None
        self.master=None

    def read_sta(self,  conf_file):

        self.input_data=pd.read_csv(conf_file['AFI']['local_data'] + '/'
                                    + conf_file['AFI']['input_data'])
        self.pre_predictions=pd.read_csv(conf_file['AFI']['local_data'] + '/'
                            + conf_file['AFI']['pre_predictions'])
        self.master=pd.read_csv(conf_file['AFI']['local_data'] + '/'
                            + conf_file['AFI']['master'])
        return

    def read_pred(self, conf_file):
        self.pred= pd.read_csv(conf_file['AFI']['local_data'] + '/'
                                    + conf_file['AFI']['predictions'])
        return

    def initial_training(self):

        train_data, test_data = train_test_split(self.input_data, shuffle =True, test_size =.05)
        colname_e = 'e'
        colname_t = 't'
        surv_train = survival_df(train_data, t_col=colname_t, e_col=colname_e, label_col="Y")
        surv_test = survival_df(test_data, t_col=colname_t, e_col=colname_e, label_col="Y")
        # Number of features in your dataset
        input_nodes = len(surv_train.columns) - 1
        # Specify your neural network structure
        hidden_layers_nodes = [6, 3, 1]
        # the arguments of dsnn can be obtained by Bayesian Hyperparameters Tuning.
        # It would affect your model performance largely!
        nn_config = {
            "learning_rate": 0.07,
            "learning_rate_decay": 1.0,
            "activation": 'tanh',
            "L1_reg": 3.4e-5,
            "L2_reg": 8.8e-5,
            "optimizer": 'sgd',
            "dropout_keep_prob": 1.0,
            "seed": 1
        }
        # ESSENTIAL STEP-1: Pass arguments
        self.model = dsnn(
            input_nodes,
            hidden_layers_nodes,
            nn_config
        )

        # ESSENTIAL STEP-2: Build Computation Graph
        self.model.build_graph()
        Y_col = ["Y"]
        X_cols = [c for c in surv_train.columns if c not in Y_col]


        self.watch_list = self.model.train(
            surv_train[X_cols], surv_train[Y_col],
            num_steps=2000,
            num_skip_steps=100,
            plot=False,
            load_model='../model/afe')

        print("CI on training data:", self.model.evals(surv_train[X_cols], surv_train[Y_col]))
        print("CI on test data:", self.model.evals(surv_test[X_cols], surv_test[Y_col]))
        #stest=surv_test[X_cols]
        pred=self.model.predict_survival_function(self.pre_predictions, plot=False)
        pred.to_csv('./data/predictions.csv', index=False)
        return pred

    def normalize_pred(self, df, symbols, factors):
        result = df.copy()

        for symbol in symbols:
            max_value = factors[symbol]['max']
            min_value = factors[symbol]['min']
            mean_value = factors[symbol]['mean']
            std_value = factors[symbol]['std']
            result[symbol] = (df[symbol] - mean_value) / std_value

        return result
    # def read_file(self, gcp_bucket, file_path, file_name, key_json, local):
    #     #embed()
    #     file_path = file_path + '/' + file_name
    #     local_file = local + '/' + file_name
    #     client = storage.Client.from_service_account_json(key_json)
    #     bucket = client.get_bucket(gcp_bucket)
    #     blob = bucket.blob(file_path)
    #     blob.download_to_filename(local_file)
    #     return

    # def model_prediction_tf_2(self, model_file, df, num):
    #     def coeff_determination(y_true, y_pred):
    #         from tensorflow.compat.v1.keras import backend as K
    #         SS_res = K.sum(K.square(y_true - y_pred))
    #         SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    #         return (1 - SS_res / (SS_tot + K.epsilon()))
    #     yhat=""
    #     K.clear_session()
    #
    #     if num==1:
    #
    #         self.model=load_model(model_file, custom_objects={'coeff_determination': coeff_determination})
    #         yhat=self.model.predict(df, verbose=2)
    #     elif num==2:
    #         self.model2=load_model(model_file, custom_objects={'coeff_determination': coeff_determination})
    #         yhat=self.model2.predict(df, verbose=2)
    #
    #     return yhat




    def model_predit_bulk(self, Xtest):
        # method for bulk predictions  non-probabilistic FULL

        def reverse_norm_int(Xtest, key):
            a = np.zeros(shape=Xtest.shape)
            std = self.normalize_factors[key[0]]['std']
            mean = self.normalize_factors[key[0]]['mean']
            a[:] = (Xtest[:] * std) + mean
            df = pd.DataFrame(a, columns=['pred'])

            return df



        # encode categorical

        # Xtest['AFI_Asset_Class_Grouping_enc'] = Xtest['berth1'].apply(lambda x: self.AFI_Asset_Class_Grouping_enc.get(x))
        # Xtest['b2_enc'] = Xtest['berth2'].apply(lambda x: self.embedding_map_key2.get(x))
        # Xtest['b3_enc'] = Xtest['berth3'].apply(lambda x: self.embedding_map_key3.get(x))
        # Xtest['locb2_enc'] = Xtest['locb2'].apply(lambda x: self.locb2.get(x))
        # Xtest['journey_id_enc'] = Xtest['journey_id'].apply(lambda x: self.journey_id_dicc.get(x))
        # Xtest['train_uid_enc'] = Xtest['train_uid'].apply(lambda x: self.train_uid_dicc.get(x))
        # Xtest['origintiploc_enc'] = Xtest['origintiploc'].apply(lambda x: self.ori_dicc.get(x))
        # Xtest['desttiploc_enc'] = Xtest['desttiploc'].apply(lambda x: self.dest_dicc.get(x))
        # Xtest['weather_main_enc'] = Xtest['weather_main'].apply(lambda x: self.wheather_main.get(x))
        # Xtest['weather_description_enc'] = Xtest['weather_description'].apply(lambda x: self.wheather_desc.get(x))
        # Xtest['season_enc'] = Xtest['season'].apply(lambda x: self.season.get(x))

        cols=['Asset_Number']
        Xfinal=Xtest[cols]
        df = self.normalize_pred(Xfinal,cols,self.AFI_factors_norm)
        #embed()
        cols2=list(self.pred.columns)

        ncols = []
        for col in cols2:
            ncols.append(str(np.int(np.float(col) * 24 // 1)))
        self.pred.columns=ncols
        Xfinal2 = pd.concat([Xfinal,self.pred[['1']]], ignore_index=True, axis=1)
        Xfinal2.columns=['Asset_Number', 'prob_survival']

        Xfinal2 = pd.merge(Xfinal2, self.master[['Asset_Number','ELR','Asset_Class_Grouping']], on=['Asset_Number'])

        return Xfinal2.to_json(orient='index')#, index=False)



    def load_dictionaries(self, conf_file):
        def load_pickle(file_input):
            with open(file_input, "rb") as input_file:
                file = pickle.load(input_file)
            return file

        #AFI_Asset_Class_Grouping_enc

        self.AFI_Asset_Class_Grouping_enc=load_pickle(conf_file['AFI']['local_conf'] + '/'
                                   + conf_file['AFI']['dictionaries']['AFI_Asset_Class_Grouping_enc'])

        #TRAIN_UID DICTIONARY

        self.AFI_ELR_enc=load_pickle(conf_file['AFI']['local_conf']  + '/'
                                + conf_file['AFI']['dictionaries']['AFI_ELR_enc'])

        #ORIGINTIPLOC DICTIONARY

        self.AFI_Engineering_Suffix_enc=load_pickle(conf_file['AFI']['local_conf']  + '/'
                                + conf_file['AFI']['dictionaries']['AFI_Engineering_Suffix_enc'])

        #DESTIPLOC DICTIONARY

        self.AFI_Grouping_Full_Name_enc=load_pickle(conf_file['AFI']['local_conf']  + '/'
                                + conf_file['AFI']['dictionaries']['AFI_Grouping_Full_Name_enc'])

        #statistics DICTIONARY

        self.AFI_HLC_enc=pd.read_pickle(conf_file['AFI']['local_conf']  + '/'
                                + conf_file['AFI']['dictionaries']['AFI_HLC_enc'])

        # Berth  distances DICTIONARY

        self.AFI_System_Asset_Type_enc=pd.read_pickle(conf_file['AFI']['local_conf']  + '/'
                                + conf_file['AFI']['dictionaries']['AFI_System_Asset_Type_enc'])

        #KEYS DICTIONARY berth1

        self.AFI_weather_station_enc=load_pickle(conf_file['AFI']['local_conf']  + '/'
                                + conf_file['AFI']['dictionaries']['AFI_weather_station_enc'])

        #KEYS DICTIONARY berth2

        self.EQUIP_CLASS_DESC_enc=load_pickle(conf_file['AFI']['local_conf'] + '/'
                                + conf_file['AFI']['dictionaries']['EQUIP_CLASS_DESC_enc'])


        #KEYS DICTIONARY berth3

        self.AFI_factors_norm=load_pickle(conf_file['AFI']['local_conf'] + '/'
                                            + conf_file['AFI']['dictionaries']['AFI_factors_norm'])


        return

