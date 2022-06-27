import json
import os

def is_partial_utterance(utt):
    if "DisplayText" in utt:
        return False
    return True


def style(n):
    n = str(n)
    n_new = n    
    for i in range(4-len(n)):
        n_new = "0" + n_new
    return n_new


def convert_to_seconds(value):
    value = int(value)
    return value/(10000000.0)


def save_data(data, folder, audio_name, channel, cont, partials):

    if data["DisplayText"] == "":
        return

    start = convert_to_seconds(data["Offset"])
    duration = convert_to_seconds(data["Duration"])
    end = start + duration

    transcription_name = os.path.join(folder, "sw" + audio_name + channel + "-ms98-a-trans.text")

    first_column = ""
    with open(transcription_name, 'a') as of:
        first_column = "sw" + audio_name + channel + "-ms98-a-" + style(cont)
        line = first_column + " " + "{:.6f}".format(start) + " " +  "{:.6f}".format(end) + " " + data["NBest"][0]["Lexical"]
        #["DisplayText"]#["NBest"][0]["Lexical"]
        print(line, file=of)

    partial_transcription_name = os.path.join(folder, "sw" + audio_name + channel + "-partial.text")
    with open(partial_transcription_name, 'a') as of:
        print(json.dumps({first_column: partials}), file=of)

    '''
    word_name = os.path.join(folder, "sw" + audio_name + channel + "-ms98-a-word.text")

    with open(word_name, 'a') as of:
        for word in data["NBest"][0]["Words"]:
            start = convert_to_seconds(word["Offset"])
            duration = convert_to_seconds(word["Duration"])
            end = start + duration
            line = first_column + "      " + "{:.6f}".format(start) + "      " +  "{:.6f}".format(end) + "     " + word["Word"]
            print(line, file=of)
    '''


def read_asr_json_transcription(fname, folder, audio_name, channel):
    cont = 1
    with open(fname, "r", encoding="utf8") as f:
        partials = []
        for line in f:
            recognized_utt = json.loads(line)
            if not is_partial_utterance(recognized_utt):
                save_data(recognized_utt, folder, audio_name, channel, cont, partials)
                partials = []
                cont += 1
            else:
                if "Text" in recognized_utt:
                    partials.append(recognized_utt["Text"]) #recognized_utt["Text"]


def generate_transcriptions(path):
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()

            folder = line[:2]
            output = os.path.join("dataset/switchboard_word_alignments/swb_ms98_transcriptions_asr_lexical", folder, line)
            try:
                os.makedirs(output)                
            except:
                pass

            for channel in ["A", "B"]:
                try:
                    fname = os.path.join("dataset/switchboard_word_alignments/swb_ms98_transcriptions_asr_json", folder, line, "sw" + line + channel + ".transcript")
                    print("Reading", fname)
                    read_asr_json_transcription(fname, output, line, channel)
                except:
                    print("Error reading", fname)
                    pass


#output = "dataset/switchboard_word_alignments/swb_ms98_transcriptions_asr_lexical/46/4637"
#fname ="dataset/switchboard_word_alignments/swb_ms98_transcriptions_asr_json/46/4637/sw4637B.transcript"
#read_asr_json_transcription(fname, output, "4637", "B")


generate_transcriptions("dataset/split/val.txt")




