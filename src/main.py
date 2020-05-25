import argparse
from utils.utils import load_conf_file, setup_logging
from dataprep.create_datasets import main_create_datasets
from dataprep.raw_to_formatted_data import main_raw_data_to_formatted_data
from modelling import TfidfGbmManager, LSTMManager, BowGbmNaNcAccidentManager, BowGbmNaNcConsoManager

########################################################################################################################
# Parsing args
########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--search", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--prod", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--dataprep", action="store_true")
parser.add_argument("--model", help="name of the model as it is in conf file", type=str)
parser.add_argument("--pathconf", default="conf/conf.json")
args = parser.parse_args()

########################################################################################################################
# Load conf
########################################################################################################################
conf = load_conf_file(args.pathconf)

########################################################################################################################
# settup logging
########################################################################################################################
setup_logging()

########################################################################################################################
# Main
########################################################################################################################
if __name__ == '__main__':
    if args.dataprep:
        main_raw_data_to_formatted_data(conf)
        main_create_datasets(conf)

    if args.model == "tfidf_gbm":
        manager = TfidfGbmManager(args.model, conf, args.train, args.search, args.eval, args.prod)
        manager.run()

    elif args.model == "lstm":
        manager = LSTMManager(args.model, conf, args.train, args.search, args.eval, args.prod)
        manager.run()

    elif args.model == "nanc_acc_bowgbm":
        manager = BowGbmNaNcAccidentManager(args.model, conf, args.train, args.search, args.eval, args.prod)
        manager.run()

    elif args.model == "nanc_conso_bowgbm":
        manager = BowGbmNaNcConsoManager(args.model, conf, args.train, args.search, args.eval, args.prod)
        manager.run()
