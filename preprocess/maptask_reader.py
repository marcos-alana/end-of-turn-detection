import xml.etree.ElementTree as ET
import re

# path 'timed-units/q1ec1.f.timed-units.xml'


MAPTASK_WORD_MAP = {
    "mm-hmm": "mhm",
    "mm-hm": "mhm",
    "mmhmm": "mhm",
    "mm-mm": "mmm",
    "uh-huh": "uhuh",
    "uh-uh": "uhuh",
    "right-o": "righto",
}

BACKCHANNELS = [
    "huh-uh",
    "hum-um",
    "um",
    "oh really",
    "oh uh-huh",
    "oh yeah",
    "oh",
    "right",
    "uh-huh uh-huh",
    "uh-huh yeah",
    "um-hum um-hum",
    "uh-huh",
    "uh-hum",
    "uh-hums",
    "uh-oh",
    "um-hum",
    "yeah yeah",
    "yeah",
    "mmhmm",
    "mhm",
    "mmm",
    "uhuh"
]

def clean_maptask_word(w):
    w = MAPTASK_WORD_MAP.get(w, w)
    w = re.sub('"', "", w)  # remove quotations
    w = re.sub("^'", "", w)  # remove "'" at the start ("'bout", "'cause")
    w = re.sub("'$", "", w)  # remove "'" at the end ("apaches'")
    w = re.sub("\-\-+", "", w)
    w = re.sub("\-\w(\-\w)+", "", w)  # na-a-a-a-a -> na
    return w

def is_longer_backchannel(words, i):
    left = True
    right = True
    if i-1 >= 0:
        left = ((float(words[i]["start"]) - words[i-1]["end"]) > 1.0) or words[i-1]["text"] in BACKCHANNELS

    if i + 1 < len(words):
        right = ((words[i+1]["start"] - float(words[i]["end"])) > 1.0) or words[i+1]["text"] in BACKCHANNELS

    if left and right:
        return True
    return False


def extract_filter_utterances_by_word_level_annotation(session_id, speaker):

    tree = ET.parse(session_id + "." + str(speaker) + ".timed-units.xml")
    root = tree.getroot()

    speaker = 0 if speaker == "f" else 1

    words = []
    for child in root:
        if str(child.attrib.get("utt")).isnumeric():
            _id = int(child.attrib.get("utt"))
            word = {"id": _id, "speaker": speaker, "text": clean_maptask_word(child.text), 
                "start": float(child.attrib.get("start")), "end": float(child.attrib.get("end"))}
            words.append(word)

    # Filtering out BACKCHANNELS
    final_words = []
    for i, word in enumerate(words):
        if word["text"] in BACKCHANNELS:
            if not is_longer_backchannel(words, i):
                final_words.append(word)
        else:
            final_words.append(word)
    return final_words




def extract_utterances_by_word_level_annotation(session_id, speaker):

    tree = ET.parse(session_id + "." + str(speaker) + ".timed-units.xml")
    root = tree.getroot()

    speaker = 0 if speaker == "f" else 1

    tmp_utts = {}
    for child in root:
        if str(child.attrib.get("utt")).isnumeric():
            _id = int(child.attrib.get("utt"))
            words = []
            if _id not in tmp_utts:
                tmp_utts[_id] = []
            else:
                words = tmp_utts[_id]
            word = {"text": child.text, "start": float(child.attrib.get("start")), "end": float(child.attrib.get("end"))}
            words.append(word)
            tmp_utts[_id] = words

    utterances = []
    for tmp_utt in tmp_utts:
        _id = tmp_utt
        text = " ".join([w["text"] for w in tmp_utts[_id]])
        start = tmp_utts[_id][0]["start"]
        end = tmp_utts[_id][-1]["end"]
        words = tmp_utts[_id]
        utterances.append({"id": _id, "speaker": speaker, "text": text, "words": words, "start": start, "end": end})

    return utterances

def extract_dialog(session):

    utterances_0 = extract_utterances_by_word_level_annotation(session, "f")
    utterances_1 = extract_utterances_by_word_level_annotation(session, "g")

    dialog = utterances_0 + utterances_1
    dialog.sort(key=lambda x: x["start"])
    return dialog, utterances_0, utterances_1


def extract_dialog_by_words(session):

    utterances_0 = extract_filter_utterances_by_word_level_annotation(session, "f")
    utterances_1 = extract_filter_utterances_by_word_level_annotation(session, "g")

    dialog = utterances_0 + utterances_1
    dialog.sort(key=lambda x: x["start"])
    return dialog, utterances_0, utterances_1




