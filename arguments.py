""" File to hold arguments """
import argparse

# data arguments

parser = argparse.ArgumentParser(description="Main Arguments")

parser.add_argument(
  '-input-data', '--input_data', type=str, required=False, help='Path to dataset')

parser.add_argument(
  '-train-data', '--train_data', type=str, required=False, help='Path to train dataset')

parser.add_argument(
  '-dev-data', '--dev_data', type=str,  required=False, help='Path to dev dataset')

parser.add_argument(
  '-test-data', '--test_data', type=str, required=False, help='Path to test dataset')

# training parameters
parser.add_argument(
  '-epochs', '--epochs', type=int, required=False, default=3, help='Number of training epochs')

parser.add_argument(
  '-batch-size', '--batch_size', type=int, required=False, default=8, help='Batch size')
parser.add_argument(
  '-max-length', '--max_length', type=int, required=False, default=256, help='Max length in encoder')

parser.add_argument(
  '-lr','--learning_rate', type=float, required=False, default=3e-5, help='Learning rate')

parser.add_argument(
  '-grad-accum', '--grad_accum', type=int, default=2, required=False, help='Gradient accumulation. Default: 2')

parser.add_argument(
  '-seed', '--seed', type=int, default=32, required=False, help='Seed')

parser.add_argument(
  '-gpu','--gpu', action='store_true', required=False, help='Use GPU or CPU')

parser.add_argument(
  '-weight-regular-token','--weight_regular_token', type=float, required=False, default=1.0, help='Regular weight')

parser.add_argument(
  '-weight-eot-token','--weight_eot_token', type=float, required=False, default=1.0, help='End-of-turn weight')

parser.add_argument(
  '-model', '--model', default='gpt2', type=str, required=False, help='Pretrained model to be used')

parser.add_argument(
  '-tokenizer', '--tokenizer', default='gpt2', type=str, choices=['gpt2'], required=False, help='Pretrained model to be used')

parser.add_argument(
  '-criteria', '--criteria', default='bAcc', type=str, choices=['bAcc', 'f1', 'f1-eot'], required=False, help='Criteria for determining the threshold. Used with -find-best')


parser.add_argument(
  '-num-utt-per-dialog', '--num_utt_per_dialog', type=int, default=1, required=False, help='Number of utterances to be regarded in the input. Default: 1')

parser.add_argument(
  '-context-size', '--context_size', type=int, required=False, default=-1, help='Context size')

parser.add_argument(
  '-output', '--output', type=str, required=False, default="", help='Output folder')

parser.add_argument(
  '-error-file', '--error_file', type=str, required=False, default="", help='Error file')

parser.add_argument(
  '-threshold','--threshold', type=float, required=False, default=-.1, help='Threshold for evaluating')

parser.add_argument(
  '-find-best','--find_best', action='store_true', required=False, help='Find the best threshold and return the performance')

def get_args():
    args = parser.parse_args()
    return args
