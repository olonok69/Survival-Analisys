import pandas as pd
pd.options.mode.chained_assignment = None
import sys
from server import run_server
from app import create_app
from IPython import embed
import argparse
import warnings
from afi import afi_estimator
import constans as c
warnings.filterwarnings('ignore')

import os
import yaml
import logging
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger('afi')
java_logger = logging.getLogger('py4j')
java_logger.setLevel(logging.INFO)


TRAIN='train'
PREDICTION='prediction'

def parse_args(args):
    """
    Parse command line parameters. Primary entry point is `etl`.
    Sub-parsers are used for each of it's specific commands.

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description='A Python interface for survival Analisys.')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    # create the parser for the "cp" command (to copy files to HDFS via
    # command line)

    parser_prediction = subparsers.add_parser(
        PREDICTION, help='Prediction command. Example use:\n'
                         '$ python runner.py prediction -y conf/data.yaml -t BRT.  '
                         'Yaml key "train_file" contais appropiate train file prepared in create section')

    parser_prediction.add_argument(
        '-t', '--type',
        dest='pred_type',
        type=str,
        help='A valid prediction type must be given.'
             'Prediction type AFI(Asset failure interface),'
             'For an example see: '
             '` python runner.py prediction -y conf/data.yaml -t BRT`.',
        required = True
    )

    parser_prediction.add_argument(
        '-y', '--yaml',
        dest='conf_file',
        type=str,
        help='YAML configuration file',
        required = True
    )
    parser_prediction = subparsers.add_parser(
        TRAIN, help='Prediction command. Example use:\n'
                         '$ python runner.py train -y conf/data.yaml -t AFI.  '
                         'Yaml key "train_file" contais appropiate train file prepared in create section')

    parser_prediction.add_argument(
        '-t', '--type',
        dest='pred_type',
        type=str,
        help='A valid prediction type must be given.'
             'Prediction type AFI(Asset failure interface),'
             'For an example see: '
             '` python runner.py prediction -y conf/data.yaml -t BRT`.',
        required=True
    )

    parser_prediction.add_argument(
        '-y', '--yaml',
        dest='conf_file',
        type=str,
        help='YAML configuration file',
        required=True
    )

    return parser.parse_args(args)

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """

    def load_yaml(file):
        with open(file, 'r') as stream:
            data = yaml.load(stream, Loader=yaml.FullLoader)

        return data

    args = parse_args(args)
    mode = args.command
    RELATIVE_YAML_FILENAME = args.conf_file

    RELATIVE_YAML_FILENAME = os.path.join(
        c.RAIL_ROOT, RELATIVE_YAML_FILENAME)



    if mode == TRAIN:
        if args.pred_type not in ['AFI']:

            raise ValueError('A valid prediction type must be given.'
                             'Prediction type BRT(Berth running time model),'
                             'For an example see: '
                             '` python runner.py prediction -y conf/data.yaml -t BRT`.')


        if args.conf_file is None:
            raise ValueError('A Yaml Configuration file  must be given. For an '
                             'example see: '
                             '`python runner.py prediction -y conf/data.yaml -t BRT`.')

        conf_file = load_yaml(RELATIVE_YAML_FILENAME)
        # Required Parameters
        PROJECT_ID = conf_file['AFI']['project_google']

        DEPLOYMENT_NAME = conf_file['AFI']['bucket_google']
        BUCKET_NAME = conf_file['AFI']['bucket_google']  # '{}-{}-bucket'.format(PROJECT_ID,DEPLOYMENT_NAME)
        key_json = conf_file['AFI']['key_json']
        if args.pred_type == 'AFI':

            new_afi = afi_estimator()
            new_afi.load_dictionaries(conf_file)

            logger.info("TPB loading training Dataset: ")
            # specify the colnames of observed status and time in your dataset
            new_afi.read_sta(conf_file)
            new_afi.pred = new_afi.initial_training()
        return
    elif mode == PREDICTION:
        if args.pred_type not in ['AFI']:
            raise ValueError('A valid prediction type must be given.'
                             'Prediction type BRT(Berth running time model),'
                             'For an example see: '
                             '` python runner.py prediction -y conf/data.yaml -t BRT`.')

        if args.conf_file is None:
            raise ValueError('A Yaml Configuration file  must be given. For an '
                             'example see: '
                             '`python runner.py prediction -y conf/data.yaml -t BRT`.')

        conf_file = load_yaml(RELATIVE_YAML_FILENAME)
        if args.pred_type == 'AFI':

            app = create_app( 'AFI', conf_file, args)
            run_server(app)

        return
def run():
    """
    Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == '__main__':
    run()