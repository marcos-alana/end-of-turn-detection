import os
from switchboard import SwitchboardUtils
import json
import pandas as pd

'''
DATASET_PATH = "dataset"

dialog_splits = {"train": [], "val": [], "test": []}

## Load split names
with open(os.path.join(DATASET_PATH, "split", "train.txt")) as f:
    session_train = [line.strip() for line in f]

with open(os.path.join(DATASET_PATH, "split", "val.txt")) as f:
    session_val = [line.strip() for line in f]

with open(os.path.join(DATASET_PATH, "split", "test.txt")) as f:
    session_test = [line.strip() for line in f]

def is_overlap(current, prev):
    start_within = prev["start"] <= current["start"] <= prev["end"]
    return start_within
'''

def build_candidate_ipus(words, pause=0.5):

    candidate_ipus = []
    ipu = [words[0]]
    for word in words[1:]:
        if word["start"] - ipu[-1]["end"] < pause:
            ipu.append(word)
        else:
            candidate_ipus.append(ipu)
            ipu = [word]
    if len(ipu) > 0:
        candidate_ipus.append(ipu)
    return candidate_ipus


def create_turns(candidate_ipus, context=1):

    stats = {"empty":0, "non-empty":0}
    empty_turn = {"text":"[empty]", "speaker":-1}
    ipus = [empty_turn]
    for cand in candidate_ipus:
        text = " ".join([token["text"] for token in cand])
        speakers = list(set([token["speaker"] for token in cand]))
        if len(speakers) > 1:
            ipus.append(empty_turn)
            stats["empty"] += 1
        else:
            speaker_id = 0 if speakers[0] == "A" else 1
            ipus.append({"text": text, "speaker": speaker_id})
            stats["non-empty"] += 1

    #merging ipus into turns
    turns = [ipus[0]]
    last_speaker = ipus[0]["speaker"]    
    for ipu in ipus[1:]:
        if ipu["speaker"] == last_speaker:
            if last_speaker != -1:
                turns[-1]["text"] += " " + ipu["text"]
            else:
                turns.append(ipu)
        else:
            turns.append(ipu)
            last_speaker = ipu["speaker"]

    #generating input turns according to a context
    instance_turns = [empty_turn for _ in range(context)]
    dataset = []

    for index, turn in enumerate(turns):
        if turn["speaker"] != -1:
            instance_turns.append(turn)
        else:
            instance_turns.append(empty_turn)

        if len(instance_turns) == context + 1:

            if len([inst["speaker"] for inst in instance_turns if inst["speaker"] == -1]) != context + 1 and \
                instance_turns[-1]["speaker"] != -1:
                current_row = [inst for inst in instance_turns]
                dataset.append(current_row)

            instance_turns.pop(0)
            
    return dataset, stats


def create_turns_by_ipus(candidate_ipus, context=1):

    stats = {"empty":0, "non-empty":0}
    empty_turn = {"text":"[empty]", "speaker":-1}
    ipus = [empty_turn]
    for cand in candidate_ipus:
        text = " ".join([token["text"] for token in cand])
        speakers = list(set([token["speaker"] for token in cand]))
        if len(speakers) > 1:
            ipus.append(empty_turn)
            stats["empty"] += 1
        else:
            #speaker_id = 0 if speakers[0] == "A" else 1 FOR SWB CORPUS
            #ipus.append({"text": text, "speaker": speaker_id})  FOR SWB CORPUS
            ipus.append({"text": text, "speaker": speakers[0]})  #FOR MAPTASK CORPUS
            stats["non-empty"] += 1

    #merging ipus into turns
    turns = [[ipus[0]]]
    last_speaker = ipus[0]["speaker"]    
    for ipu in ipus[1:]:
        if ipu["speaker"] == last_speaker:
            if last_speaker != -1:
                turns[-1].append(ipu)
            else:
                turns.append([ipu])
        else:
            turns.append([ipu])
            last_speaker = ipu["speaker"]

    #generating input turns according to a context
    instance_turns = [[empty_turn] for _ in range(context)]
    dataset = []

    for index, turn in enumerate(turns):

        if turn[0]["speaker"] != -1:
            instance_turns.append(turn)
        else:
            instance_turns.append([empty_turn])

        #print(instance_turns)

        if len(instance_turns) == context + 1:

            if len([inst[0]["speaker"] for inst in instance_turns if inst[0]["speaker"] == -1]) != context + 1 and \
                instance_turns[-1][0]["speaker"] != -1:

                context_row = []
                for inst in instance_turns[:len(instance_turns)-1]:
                    text = " ".join([partial["text"] for partial in inst])
                    speaker = inst[0]["speaker"]
                    context_row.append({"text": text, "speaker": speaker})

                current_text = []
                for idx, partial in enumerate(instance_turns[len(instance_turns)-1]):
                    current_text += [partial["text"]]
                    if idx + 1 == len(instance_turns[len(instance_turns)-1]):
                        dataset.append(context_row + [{"text": " ".join(current_text), "speaker": 1}])
                    else:
                        dataset.append(context_row + [{"text": " ".join(current_text), "speaker": 0}])

            instance_turns.pop(0)
            
    return dataset, stats


def generate_ipus(candidate_ipus):

    generated_ipus = []    
    for cand in candidate_ipus:
        new_ipu = [cand[0]]
        for word in cand[1:]:
            if word["speaker"] == new_ipu[-1]["speaker"]:
                new_ipu += [word]
            else:
                for idx, e in enumerate(reversed(new_ipu)):
                    if not is_overlap(word, e):
                        break
                if idx > 0:
                    del new_ipu[-(idx):]
                duration = new_ipu[-1]["end"] - new_ipu[0]["start"]
                if duration >= 1.0:
                    generated_ipus.append(new_ipu)
                new_ipu = [word]
        duration = new_ipu[-1]["end"] - new_ipu[0]["start"]
        if duration >= 1.0:
            generated_ipus.append(new_ipu)

    turns = [generated_ipus[0]]
    last_speaker = generated_ipus[0][0]["speaker"]
    for ipu in generated_ipus[1:]:
        if ipu[0]["speaker"] == last_speaker:
            turns[-1] += ipu
        else:
            turns.append(ipu)
            last_speaker = ipu[0]["speaker"]

        utterance = " ".join([w["text"] for w in ipu])
        speaker = ipu[0]["speaker"]

    return turns
'''        
total_utterances = 0
total_refined_utterances = 0

pause = 500
context = 1

transcription_folder = os.path.join(DATASET_PATH, "switchboard_word_alignments", "swb_ms98_transcriptions")
empty = 0
non_empty = 0
instances = 0

for idx1, folder in enumerate(os.listdir(transcription_folder)):

    if os.path.isdir(os.path.join(transcription_folder, folder)):
        for idx2, session in enumerate(os.listdir(os.path.join(transcription_folder, folder))):

            root = os.path.join(transcription_folder, folder, session.strip())
            print(root, "\t", session)
            #print(folder, session)
            
            if session in session_val:
                ds = "val"
            elif session in session_test:
                ds = "test"
            else:
                ds = "train"

            f_a = os.path.join(root, "sw" + session + "A-ms98-a-word.text")
            words_a = SwitchboardUtils.extract_ipus_from_word_level_annotations(session, "A", root, apply_regexp=True)

            f_b = os.path.join(root, "sw" + session + "B-ms98-a-word.text")
            words_b = SwitchboardUtils.extract_ipus_from_word_level_annotations(session, "B", root, apply_regexp=True)
            words = words_a + words_b
            words.sort(key=lambda x: x["start"])

            candidate_ipus = build_candidate_ipus(words, pause/1000)
            dataset, stats = create_turns_by_ipus(candidate_ipus, context=context)

            #turns = generate_ipus(candidate_ipus)
            instances += len(dataset)

            empty += stats["empty"]
            non_empty += stats["non-empty"]

            dialog = []
            for turns in dataset:
                dialog.append([inst[key] for inst in turns for key in inst])

            dialogs = dialog_splits[ds]
            dialogs += dialog
            dialog_splits[ds] = dialogs


for key in dialog_splits:
    print(key, "\t", len(dialog_splits[key]))

    df = pd.DataFrame(dialog_splits[key])
    df.to_csv(os.path.join(DATASET_PATH, "skantze-ipu", str(pause) + "ms", key + '.csv'), encoding="utf-8", sep="\t", index=False)

print(empty, "\t", non_empty, "\t", instances)
'''
