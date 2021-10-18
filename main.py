import yaml
import datetime
import argparse
import configparser
import logging.config
from util.dataset import load_dataset
from util.train import set_model
from util.classify import test



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    args = arg_parser.parse_args()

    with open("./bin/logger.yaml", "rt") as file:
        logger_config = yaml.safe_load(file.read())
    logging.config.dictConfig(logger_config)
    logger = logging.getLogger("default")
    logger.info(" ~~~~~~~{date_} process START".format(date_=datetime.datetime.today().strftime('%Y-%m-%d')))

    config = configparser.ConfigParser()
    config.read("./bin/config.conf")

    data_path, aws_info, model_info = config["DATA_PATH"], config["AWS_INFO"], config["MODEL_INFO"]
    data_path_dict, aws_info_dict, model_info_dict = dict([(k, v) for k, v in data_path.items()]), \
                                                     dict([(k, v) for k, v in aws_info.items()]),\
                                                     dict([(k, v) for k, v in model_info.items()])


    logger.info("   1. Load Dataset")
    load_dataset(aws_info_dict['accesskey'], aws_info_dict['secretkey'], aws_info_dict['bucket_name'], data_path_dict['dataset'])


    logger.info("   2. Train Dataset")
    all_label = set(['outerwear', 'tops', 'pants', 'dresses'])
    set_model = set_model()

    set_model.parameter(data_path_dict['training'], data_path_dict['model'], data_path_dict['plot'],
                        model_info_dict['epochs'], model_info_dict['init_lr'], model_info_dict['bs'], all_label)
    batch_generator = set_model.batch_generator()
    set_model.iter_train(batch_generator)


    logger.info("   3. Test Dataset")
    test(data_path_dict['model'], data_path_dict['validation'], all_label)


