from flask import Blueprint,Flask, request, Response, jsonify, json
main = Blueprint('main', __name__)
import pandas as pd
import constans as c
from IPython import embed
import os
import sys
import datetime
from afi import afi_estimator
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/afi/predict", methods=["POST", "GET"])
def bulk_runtimea():
    # full only 1 row
    b1= datetime.datetime.now().timestamp()
    submitted_file = request.get_json()
    X_test=pd.io.json.json_normalize(submitted_file,'data')
    X_test.columns=c.json_source_reduced
    dicc = estimator_prob.model_predit_bulk(X_test)

    b2=datetime.datetime.now().timestamp()
    timeb=b2-b1
    logger.info("Request time ...: {}".format(timeb))
    logger.info("prediction: {}".format(dicc))
    return jsonify(dicc)



def create_app( type_object, conf_file, args):

    global estimator_prob

    if type_object =='AFI':
        estimator_prob=_create_objects(conf_file, args)


    app = Flask(__name__)
    app.register_blueprint(main)
    return app


def _create_objects(conf_file, args):

    new_afi = afi_estimator()
    new_afi.load_dictionaries(conf_file)

    logger.info("TPB loading training Dataset: ")
    # specify the colnames of observed status and time in your dataset
    new_afi.read_sta(conf_file)
    new_afi.read_pred(conf_file)
    #new_afi.pred= new_afi.initial_training()

    return new_afi