from util import read_txt, load_partial
import re
from os import walk
from os.path import join, basename
from copy import deepcopy


OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]

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
]


BACKCHANNEL_MAP = {
    "uh-huh": "uhuh",
    "huh-uh": "uhuh",
    "uh-hum": "mhm",
    "uh-hums": "mhm",
    "um-hum": "mhm",
    "hum-um": "mhm",
    "uh-oh": "uhoh",
}


def swb_regexp(s):
    """
    Switchboard annotation specific regexp.
    See:
        - https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf
    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


class SwitchboardUtils:
    @staticmethod
    def extract_word_level_annotations(session_id, speaker, root, apply_regexp=False):
        def remove_multiple_whitespace(s):
            s = re.sub(r"\t", " ", s)
            return re.sub(r"\s\s+", " ", s)

        # Load word-level annotations
        words_filename = "sw" + session_id.strip() + speaker.strip() + "-ms98-a-word.text"
        words_list = read_txt(join(root, words_filename))

        # process word-level annotation
        word_dict = {}
        for word_row in words_list:
            word_row = remove_multiple_whitespace(word_row).strip()
            try:
                idx, wstart, wend, word = word_row.split(" ")
            except Exception as e:
                print("word_row: ", word_row)
                print("word_split: ", word_row.split(" "))
                print(e)
                input()
                assert False

            if apply_regexp:
                word = swb_regexp(word)

            if not (word in OmitText or word == ""):
                if idx in word_dict:
                    word_dict[idx].append(
                        {"text": word, "start": float(wstart), "end": float(wend)}
                    )
                else:
                    word_dict[idx] = [
                        {"text": word, "start": float(wstart), "end": float(wend)}
                    ]
        return word_dict


    @staticmethod
    def extract_ipus_from_word_level_annotations(session_id, speaker, root, apply_regexp=True, pause=.5):
        def remove_multiple_whitespace(s):
            s = re.sub(r"\t", " ", s)
            return re.sub(r"\s\s+", " ", s)

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

        # Load word-level annotations
        words_filename = "sw" + session_id.strip() + speaker.strip() + "-ms98-a-word.text"
        words_list = read_txt(join(root, words_filename))

        # process word-level annotation
        word_dict = {}
        words = []
        ipu_idx = 1
        for word_row in words_list:
            word_row = remove_multiple_whitespace(word_row).strip()
            try:
                idx, wstart, wend, word = word_row.split(" ")
            except Exception as e:
                print("word_row: ", word_row)
                print("word_split: ", word_row.split(" "))
                print(e)
                input()
                assert False

            if apply_regexp:
                word = swb_regexp(word)

            if not (word in OmitText or word == ""):                
                words.append(
                        {"id": idx, "speaker": speaker, "text": word, "start": float(wstart), "end": float(wend)}
                )

        # Filtering out BACKCHANNELS
        final_words = []
        for i, word in enumerate(words):
            if word["text"] in BACKCHANNELS:
                if not is_longer_backchannel(words, i):
                    final_words.append(word)
            else:
                final_words.append(word)

        return final_words


    @staticmethod
    def combine_speaker_utterance_and_words(
        session_id, speaker, root, apply_regexp=False, strict_times=False, partial=False
    ):
        """Combines word- and utterance-level annotations"""
        # Read word-level annotation and format appropriately
        word_dict = SwitchboardUtils.extract_word_level_annotations(
            session_id, speaker, root, apply_regexp=apply_regexp
        )

        # Read utterance-level annotation
        trans_filename = "sw" + session_id.strip() + speaker.strip() + "-ms98-a-trans.text"
        trans_list = read_txt(join(root, trans_filename))

        if partial:
            partial_filename = "sw" + session_id.strip() + speaker.strip() + "-partial.text"
            partial_list = load_partial(join(root, partial_filename))

        # correct channels for wavefiles
        speaker = 0 if speaker == "A" else 1

        # Combine word-/utterance- level annotations
        utterances = []
        for row in trans_list:
            utt_idx, start, end, *words = row.split(" ")
            try:
                if not (words[0] in OmitText and len(words) == 1):  # only noise/silence
                    wd = word_dict.get(utt_idx, None)
                    if wd is None and len(word_dict) > 0:
                        continue
                    else:
                        if partial:
                            wd = words

                    words = " ".join(words)
                    if apply_regexp:
                        words = swb_regexp(words)

                    if partial:
                        tmp_utt = {
                            "id": utt_idx,
                            "speaker": speaker,
                            "text": words,
                            "words": wd,
                            "partial": partial_list[utt_idx]
                        }
                    else:
                        #print(">>", wd)
                        tmp_utt = {
                            "id": utt_idx,
                            "speaker": speaker,
                            "text": words,
                            "words": wd
                        }

                    # if self.config.name == "default":
                    if strict_times:
                        #print(tmp_utt)
                        # Otherwise use the more exact word-level times
                        # use word start/end times for utterance
                        tmp_utt["start"] = tmp_utt["words"][0]["start"]
                        tmp_utt["end"] = tmp_utt["words"][-1]["end"]
                    else:
                        # By default use the utterance level timings (with padding)
                        # use annotated start/end times for utterance
                        tmp_utt["start"] = float(start)
                        tmp_utt["end"] = float(end)
                    utterances.append(tmp_utt)
            except:
                pass
        return utterances


    @staticmethod
    def extract_dialog(session_id, root, raw=False, partial=False):
        """Extract the annotated dialogs based on config `name`"""
        # Config settings
        apply_regexp = False
        strict_times = False
        if not raw:
            apply_regexp = True
            strict_times = True # REAL: True   MODIFIED: FALSE

        # Speaker A: original name in annotations
        a_utterances = SwitchboardUtils.combine_speaker_utterance_and_words(
            session_id,
            speaker="A",
            root=root,
            apply_regexp=apply_regexp,
            strict_times=strict_times,
            partial=partial
        )

        # Speaker B: original name in annotations
        b_utterances = SwitchboardUtils.combine_speaker_utterance_and_words(
            session_id,
            speaker="B",
            root=root,
            apply_regexp=apply_regexp,
            strict_times=strict_times,
            partial=partial
        )

        # Combine speaker utterances and sort by start-time
        dialog = a_utterances + b_utterances
        dialog.sort(key=lambda x: x["start"])
        return dialog, a_utterances, b_utterances

    @staticmethod
    def extract_vad(utterances):
        vad = [[], []]
        for utt in utterances:
            channel = utt["speaker"]
            s, e = utt["words"][0]["start"], utt["words"][0]["end"]
            for w in utt["words"][1:]:
                if w["start"] - e < 0.05:
                    e = w["end"]
                    # print('joint')
                else:
                    vad[channel].append((s, e))
                    # update
                    s = w["start"]
                    e = w["end"]
            vad[channel].append((s, e))
        return vad



class TextFocusDialog:

    def __init__(self, backchannel_list=BACKCHANNELS, lookahead_duration=2):
        self.backchannel_list = backchannel_list
        self.lookahead_duration = lookahead_duration

    def extract_context_vad(self, vad, end_time):
        """Extract vad up until `end_time`."""
        vad_end = end_time + self.lookahead_duration
        cvad = [[], []]
        for ch, ch_vad in enumerate(vad):
            for start, end in ch_vad:
                if start < vad_end:
                    if end < vad_end:
                        cvad[ch].append([start, end])
                    else:
                        cvad[ch].append([start, vad_end])
        return cvad

    def is_backchannel(self, utt):
        return utt["text"] in self.backchannel_list

    def join_utterances(self, utt1, utt2, has_partial=False):
        utt = deepcopy(utt1)
        utt["text"] += " " + utt2["text"]
        utt["words"] += utt2["words"]
        utt["end"] = utt2["end"]
        if has_partial:
            utt["partial"] += [utt1["text"] + " " + partial for partial in utt2["partial"]]
        return utt

    def is_overlap_within(self, current, prev):
        start_within = prev["start"] <= current["start"] <= prev["end"]
        end_within = prev["start"] <= current["end"] <= prev["end"]
        return start_within and end_within

    def is_overlap(self, current, prev):
        start_within = prev["start"] <= current["start"] <= prev["end"]
        end_with_no_overlap = current["end"] > prev["end"]
        return start_within and end_with_no_overlap

    def refine_dialog(self, utterances, vad=None, partial=False):
        """
        Refine the dialog by omitting `overlap_within` and `backchannel`
        speech, both of which are added to the current/major utterance. Keeps
        the original fields for text, words, start, end, speaker.
        i.e:
            refined[i] = {'id', 'speaker', 'text', 'words', 'start', 'end', 'backchannel', 'within'}
        """
        first = utterances[0]
        first["backchannel"] = []
        first["within"] = []
        first["overlap"] = []
        if vad is not None:
            first["vad"] = self.extract_context_vad(vad, first["end"])
        refined = [first]
        last_speaker = first["speaker"]
        for idx, current in enumerate(utterances[1:]):
            if self.is_backchannel(current):
                refined[-1]["backchannel"] += current["words"]
            elif self.is_overlap_within(current, refined[-1]):
                refined[-1]["within"] += current["words"]
            #elif self.is_overlap(current, refined[-1]):
            #    refined[-1]["overlap"] += current["words"]
            else:
                if current["speaker"] == last_speaker:
                    refined[-1] = self.join_utterances(refined[-1], current)
                else:
                    current["backchannel"] = []
                    current["within"] = []
                    current["overlap"] = []
                    if vad is not None:
                        current["vad"] = self.extract_context_vad(vad, current["end"])
                    refined.append(current)
                    last_speaker = current["speaker"]
        return refined


    def remove_backchannels(self, utterances):
        """
        Refine the dialog by omitting `overlap_within` and `backchannel`
        speech, both of which are added to the current/major utterance. Keeps
        the original fields for text, words, start, end, speaker.
        i.e:
            refined[i] = {'id', 'speaker', 'text', 'words', 'start', 'end', 'backchannel', 'within'}
        """

        aux_utterances = []
        for idx, current in enumerate(utterances):
            if idx == 0:
                prev = None
            else:
                prev = utterances[idx - 1]["end"]

            if idx == len(utterances) - 1:
                next = None
            else:
                next = utterances[idx + 1]["start"]

            if current["text"] in BACKCHANNELS:
                is_alone_left = True
                is_alone_right = True

                if next is not None:
                    is_alone_right = float(next) - float(current["end"]) > 1.0
                if prev is not None:
                    is_alone_left = float(current["start"]) - float(prev) > 1.0

                if not (is_alone_right and is_alone_left):
                    aux_utterances.append(current)
            else:
                aux_utterances.append(current)
        return aux_utterances


    def generate_ipus(self, utterances, ipu_threshold=0.5):
        """
        Refine the dialog by omitting `overlap_within` and `backchannel`
        speech, both of which are added to the current/major utterance. Keeps
        the original fields for text, words, start, end, speaker.
        i.e:
            refined[i] = {'id', 'speaker', 'text', 'words', 'start', 'end', 'backchannel', 'within'}
        """

        ipus = []
        ipus.append([utterances[0]])
        for idx, current in enumerate(utterances[1:]):
            if float(current["start"]) - float(ipus[-1][-1]["end"]) < ipu_threshold:
                ipus[-1] = ipus[-1] + [current]
            else:
                ipus.append([current])
        return ipus



'''
# A sorted (by start of utterance) list of utterances
session = "4024"
root = "dataset/switchboard_word_alignments/swb_ms98_transcriptions/40/4024"
utterances, a_utterances, b_utterances = SwitchboardUtils.extract_dialog(session, root)

for utt in utterances:
    print("Speaker",utt["speaker"], ":\t", utt["text"], "\t", utt["words"] , "\t,(", utt["start"] , "-", utt["end"] , ")")
print("===================")


textFocusDialog = TextFocusDialog()

refined_utterances = textFocusDialog.refine_dialog(utterances)
#print(refined_utterances)

for utt in refined_utterances:
    print("Speaker",utt["speaker"], ":\t", utt["text"], "\t", utt["backchannel"], "\t", utt["within"])

print("Original:", len(utterances))
print("Refined:", len(refined_utterances))
'''
