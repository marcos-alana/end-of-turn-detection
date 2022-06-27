import os
from maptask_reader import extract_dialog, extract_dialog_by_words
from switchboard import TextFocusDialog
import random
import json
from skantze_preprocessing import build_candidate_ipus, create_turns_by_ipus
import pandas as pd

DATASET_PATH = "dataset"

folder_path = "dataset/maptaskv2-1/timed-units"

'''
####### FOR TURNGPT SPLITS EXPERIMENTS
train_sessions = ['q7nc3', 'q1nc7', 'q6ec8', 'q7ec3', 'q4ec6', 'q5nc1', 'q3ec4', 'q3ec6', 'q5nc2', 'q7ec7', 'q3ec3', 'q1nc1', 'q2ec6', 'q7nc5', 'q7nc8', 'q5ec1', 'q6ec2', 'q8ec4', 'q8nc1', 'q5ec2', 'q4ec5', 'q3ec7', 'q4nc4', 'q5nc6', 'q3ec1', 'q2ec4', 'q5nc5', 'q5ec8', 'q4nc1', 'q2ec3', 'q8ec5', 'q6ec6', 'q1ec2', 'q6ec3', 'q7nc7', 'q7nc2', 'q6ec1', 'q3ec2', 'q6nc7', 'q2ec2', 'q1nc2', 'q1nc4', 'q3nc8', 'q3nc1', 'q1nc5', 'q7ec1', 'q5ec4', 'q2nc5', 'q1ec1', 'q1ec4', 'q8ec6', 'q2nc3', 'q4nc8', 'q5nc4', 'q4nc7', 'q5ec7', 'q7ec2', 'q4nc2', 'q2ec7', 'q6nc2', 'q8nc5', 'q2ec8', 'q8ec1', 'q6nc4', 'q1nc8', 'q6nc3', 'q4ec4', 'q7ec5', 'q8nc6', 'q5ec6', 'q7nc4', 'q7nc6', 'q8nc3', 'q4nc6', 'q2nc8', 'q7ec6', 'q5nc3', 'q8nc2', 'q3nc4', 'q3ec8', 'q1ec3', 'q4ec3', 'q5ec3', 'q2nc6', 'q6ec7', 'q2nc1', 'q4ec1', 'q1ec8', 'q3nc2', 'q4ec7', 'q8ec2', 'q7nc1', 'q3nc3', 'q2ec5', 'q2nc7', 'q1ec5', 'q4ec8', 'q4nc3', 'q8nc7', 'q4nc5', 'q5ec5', 'q4ec2', 'q6nc6', 'q3nc6', 'q2ec1', 'q2nc4', 'q1nc6', 'q3nc7', 'q6nc5', 'q8ec3', 'q8ec8', 'q3ec5', 'q6nc1', 'q5nc8', 'q8ec7', 'q6ec4']
val_sessions = ['q6nc8', 'q3nc5', 'q2nc2', 'q1ec7', 'q7ec4', 'q5nc7']
test_sessions = ['q6ec5', 'q8nc4', 'q8nc8', 'q7ec8', 'q1nc3', 'q1ec6']
'''


################# FOR SKANTZE SPLITS EXPERIMENTS

train_sessions = ['q2nc7', 'q7nc8', 'q2ec2', 'q1nc7', 'q3nc4', 'q3ec1', 'q3nc6', 'q8nc3', 'q1ec4', 'q5ec8', 'q4ec5', 'q7nc5', 'q1nc2', 'q1nc6', 'q7nc2', 'q1ec3', 'q2nc5', 'q1ec8', 'q6nc4', 'q8nc1', 'q4ec7', 'q4nc5', 'q6nc1', 'q5ec2', 'q4nc2', 'q4nc6', 'q1nc1', 'q6ec6', 'q7ec7', 'q3nc1', 'q7ec1', 'q2nc8', 'q3ec4', 'q7ec6', 'q2ec8', 'q5nc1', 'q2ec4', 'q5nc2', 'q8ec1', 'q7nc4', 'q2nc1', 'q2ec3', 'q6nc6', 'q3nc3', 'q8ec8', 'q4nc3', 'q4ec8', 'q8ec5', 'q7ec2', 'q5nc8', 'q4ec4', 'q8ec4', 'q1ec2', 'q7nc1', 'q5nc4', 'q8ec3', 'q4ec1', 'q3ec2', 'q6nc7', 'q4ec2', 'q1nc8', 'q8ec2', 'q1ec5', 'q4nc8', 'q5ec7', 'q3nc8', 'q8nc2', 'q6nc5', 'q1nc5', 'q4nc4', 'q2nc3', 'q6ec1', 'q6ec3', 'q4ec3', 'q8ec7', 'q6ec2', 'q6ec7', 'q5ec5', 'q5nc3', 'q6nc3', 'q2ec6', 'q3ec5', 'q2nc4', 'q4nc7']

val_sessions = ['q6nc8', 'q3nc5', 'q2nc2', 'q1ec7', 'q7ec4', 'q5nc7', 'q3ec3', 'q6ec8', 'q7nc7', 'q3nc2', 'q5ec1', 'q5nc5']

test_sessions = ['q6ec5', 'q8nc4', 'q8nc8', 'q7ec8', 'q1nc3', 'q1ec6', 'q5nc6', 'q6ec4', 'q2ec1', 'q3ec7', 'q8nc5', 'q3ec8', 'q7ec5', 'q6nc2', 'q8ec6', 'q2nc6', 'q4nc1', 'q5ec4', 'q2ec5', 'q3ec6', 'q3nc7', 'q5ec6', 'q8nc6', 'q2ec7', 'q8nc7', 'q1ec1', 'q7ec3', 'q7nc6', 'q1nc4', 'q4ec6', 'q7nc3', 'q5ec3']




textFocusDialog = TextFocusDialog()

dialog_splits = {"train": [], "val": [], "test": []}


def generate_dialogs(folder_path, sessions):

    dialogs = []
    for session in sessions:

        utterances, _, _ = extract_dialog(os.path.join(folder_path, session))
        refined_utterances = textFocusDialog.refine_dialog(utterances)
        dialog = []
        for utt in refined_utterances:
            dialog.append({"speaker": utt["speaker"], "text": utt["text"]})
        dialogs.append(dialog)
    return dialogs


def generate_dialogs_by_pauses(folder_path, sessions, pause, context):

    empty = 0
    non_empty = 0
    n_cand = 0

    dialogs = []
    for session in sessions:

        words, _, _ = extract_dialog_by_words(os.path.join(folder_path, session))
        candidate_ipus = build_candidate_ipus(words, pause/1000)
        dataset, stats = create_turns_by_ipus(candidate_ipus, context=context)

        empty += stats["empty"]
        non_empty += stats["non-empty"]
        n_cand += len(candidate_ipus)

        for turns in dataset:
            dialogs.append([inst[key] for inst in turns for key in inst])

    print(empty, "\t", non_empty, "\t", n_cand)
    return dialogs
'''
#pause = 500
context = 1

for pause in [50, 250, 500]:

    dialog_splits["train"] = generate_dialogs_by_pauses(folder_path, train_sessions, pause, context)
    dialog_splits["val"] = generate_dialogs_by_pauses(folder_path, val_sessions, pause, context)
    dialog_splits["test"] = generate_dialogs_by_pauses(folder_path, test_sessions, pause, context)

    for key in dialog_splits:
        print(key, "\t", len(dialog_splits[key]))

        df = pd.DataFrame(dialog_splits[key])
        #df.to_csv(os.path.join(DATASET_PATH, "skantze-ipu-maptask", str(pause) + "ms", key + '.csv'), encoding="utf-8", sep="\t", index=False)
        df.to_csv(os.path.join(DATASET_PATH, "skantze-2017", "ipu", str(pause) + "ms", key + '.csv'), encoding="utf-8", sep="\t", index=False) #FINAL EXP

'''

dialog_splits["train"] = generate_dialogs(folder_path, train_sessions)
dialog_splits["val"] = generate_dialogs(folder_path, val_sessions)
dialog_splits["test"] = generate_dialogs(folder_path, test_sessions)

for key in dialog_splits:
    print(key, "\t", len(dialog_splits[key]))
    #with open(os.path.join("maptask." + key + '.json'), 'w', encoding="UTF-8") as outfile:
    with open(os.path.join(DATASET_PATH, "skantze-2017", "lm" ,key + '.json'), 'w', encoding="UTF-8") as outfile: #FINAL EXP
        json.dump(dialog_splits[key], outfile, indent=4)

