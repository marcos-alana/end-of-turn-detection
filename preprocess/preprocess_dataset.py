import urllib.request
from zipfile import ZipFile
import tarfile
import os, os.path
from switchboard import SwitchboardUtils, TextFocusDialog
import json

_URL = "https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz"
DATASET_PATH = "dataset"
EXTENSION = ".tar.gz"
FILE_PATH = os.path.join(DATASET_PATH, os.path.basename(_URL))
FILE_PATH_NO_EXTENSION = FILE_PATH.replace(EXTENSION, "")

'''
if not os.path.isdir(DATASET_PATH):
    os.makedirs(DATASET_PATH)

if not os.path.isfile(FILE_PATH):
    print("Downloading ", _URL, "...")
    urllib.request.urlretrieve(_URL, FILE_PATH)
    print(_URL, "downloaded at", FILE_PATH)

if not os.path.isdir(FILE_PATH_NO_EXTENSION):
    print("Unziping ", FILE_PATH, "...")
    tar = tarfile.open(FILE_PATH, "r:gz")
    tar.extractall(FILE_PATH_NO_EXTENSION)
    tar.close()
    print(FILE_PATH, " unzipped")

'''
session_train = []
session_val = []

## Load split names
with open(os.path.join(DATASET_PATH, "split", "train.txt")) as f:
    session_train = [line.strip() for line in f]

with open(os.path.join(DATASET_PATH, "split", "val.txt")) as f:
    session_val = [line.strip() for line in f]

with open(os.path.join(DATASET_PATH, "split", "test.txt")) as f:
    session_test = [line.strip() for line in f]

dialog_splits = {"train": [], "val": [], "test": []}
#dialog_splits = {"val": []}

total_utterances = 0
total_refined_utterances = 0

textFocusDialog = TextFocusDialog()
transcription_folder = os.path.join(FILE_PATH_NO_EXTENSION, "swb_ms98_transcriptions")#"swb_ms98_transcriptions_asr_lexical")
for folder in os.listdir(transcription_folder):
    if os.path.isdir(os.path.join(transcription_folder, folder)):
        for session in os.listdir(os.path.join(transcription_folder, folder)):
            root = os.path.join(transcription_folder, folder, session.strip())
            print(root)
            #print(folder, session)
            
            if session in session_val:
                ds = "val"
            elif session in session_test:
                ds = "test"
            else:
                ds = "train"
            try:
                utterances, a_utterances, b_utterances = SwitchboardUtils.extract_dialog(session, root)
                total_utterances += len(utterances)
                ## Filter by interpausal-units
                refined_utterances = textFocusDialog.refine_dialog(utterances)
                total_refined_utterances += len(refined_utterances)
                dialog = []
                for utt in refined_utterances:
                    if "partial" in utt:
                        dialog.append({"speaker": utt["speaker"], "text": utt["text"], "partial": utt["partial"]})
                    else:
                        dialog.append({"speaker": utt["speaker"], "text": utt["text"]})
                dialogs = dialog_splits[ds]
                dialogs.append(dialog)
                dialog_splits[ds] = dialogs
            except:
                pass


for key in dialog_splits:
    print(key, "\t", len(dialog_splits[key]))
    with open(os.path.join(DATASET_PATH, key + '.json'), 'w', encoding="UTF-8") as outfile:
        json.dump(dialog_splits[key], outfile, indent=4)

print(total_utterances, "\t", total_refined_utterances)
