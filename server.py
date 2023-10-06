import flask
import requests
import firebase_admin
from firebase_admin import credentials, storage
from google.cloud import storage
from google.oauth2 import service_account
from pathlib import Path
import subprocess as s
import time
import glob
import os
import re
import json

feat_type = 'fbank'
cred = credentials.Certificate(
    "cgh-rcnn-flask-firebase-adminsdk-f92al-8d9038b979.json")
save_path = 'run'
Path(save_path).mkdir(parents=True, exist_ok=True)
firebase_admin.initialize_app(
    cred, {'storageBucket': 'cgh-rcnn-flask.appspot.com'})  # connecting to firebase


def download_blob(bucket_name, filename, source_blob_name):
    credentials = service_account.Credentials.from_service_account_file(
        "cgh-rcnn-flask-firebase-adminsdk-f92al-8d9038b979.json")
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(filename)
    print(f"File {source_blob_name} download to {filename}.")


def filepath_to_filename(filepath):
    tmp = os.path.basename(filepath)
    filename = os.path.splitext(tmp)[0]
    print(f'filepath: {filepath}, filename: {filename}')
    return filename


def wav_scp_prep(save_path, file_folder):
    print(save_path, file_folder)
    wav_paths = glob.glob(os.getcwd()+"/"+save_path+'/'+file_folder+"/*.wav")
    # print(os.getcwd()+"/"+save_path+file_folder+"/*.wav")
    w1 = open(save_path+"/wav.scp", 'w+')
    w2 = open(save_path+"/wav_sph.scp", 'w+')
    for wav_path in wav_paths:
        wav_name = filepath_to_filename(wav_path)
        # print(wav_name)
        utt_id = file_folder.split('_')[1]+'_'+wav_name
        w1.write(utt_id + " " + wav_path + "\n")
        w2.write(utt_id + " " + wav_path + "\n")
    w1.close()
    w2.close()


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def handle_call():
    return "Successfully Connected"


@app.route('/getfact', methods=['GET'])
def get_fact():
    return "Hey!! I'm the fact you got!!!"


@app.route('/inference/<filename>', methods=['POST'])
def extract_name(filename):
    tmp = os.path.basename(filename)
    file_folder = os.path.splitext(tmp)[0]
    print("Step1 download firbase file")
    download_blob(firebase_admin.storage.bucket().name,
                  save_path+"/"+filename, 'data_0820word/'+filename)
    print(s.run(['./unzip.sh'+' '+save_path+'/'+' '+filename], shell=True))

    print("\nStep2 prepare kaldi style data")
    wav_scp_prep(save_path, file_folder)

    print("\nStep3 data processing")
    bash_out = s.run(["./steps/make_feat.sh", feat_type, save_path], text=True)

    print("\nStep4 model inference")
    bash_out = s.run(["./steps/predict_ctc.sh"], text=True)

    print("\nStep5 doing result analysis")
    bash_out = s.run(["./steps/mdd_result.sh", save_path], text=True)
    f = open(save_path+"/ref_our_detail", 'r')
    dic = {}
    insert = 0
    delete = 0
    sub = 0
    cor = 0
    count = 0
    for line in f:
        line = line.strip()
        if ("ref" in line):
            ref = line.split("ref")
            ref[0] = ref[0].strip(" ")
            ref[1] = ref[1].strip(" ")
            ref[1] = re.sub(" +", " ", ref[1])
            ref_seq = ref[1].split(" ")
            dic[ref[0]] = []
            dic[ref[0]].append(ref[1])
        elif ("hyp" in line):
            hyp = line.split("hyp")
            hyp[0] = hyp[0].strip(" ")
            hyp[1] = hyp[1].strip(" ")
            hyp[1] = re.sub(" +", " ", hyp[1])
            hyp_seq = hyp[1].split(" ")
            dic[hyp[0]].append(hyp[1])
        elif (" op " in line):
            op = line.split(" op ")
            op[0] = op[0].strip(" ")
            op[1] = op[1].strip(" ")
            op[1] = re.sub(" +", " ", op[1])
            op_seq = op[1].split(" ")
            dic[op[0]].append(op[1])
            for i in op_seq:
                if (i == "I"):
                    insert += 1
                elif (i == "D"):
                    delete += 1
                    count += 1
                elif (i == "S"):
                    sub += 1
                    count += 1
                elif (i == "C"):
                    cor += 1
                    count += 1
    f.close()
    print("insert:", insert)
    print("delete:", delete)
    print("sub:", sub)
    print("cor:", cor)
    print("sum", count)

    Stopping = {
        "canonical_s": ["m", "f", "n", "l", "k", "kh", "x", "tɕ", "thɕ", "ɕ", "tʂ", "thʂ", "ʂ", "ʐ", "ts", "tsh", "s"],
        "perceived_s": ["p", "ph", "t", "th"],
        "number": 0
    }

    Backing = {
        "canonical_s": ["p", "ph", "m", "f", "t", "th", "n", "l", "x", "tɕ", "thɕ", "ɕ", "tʂ", "thʂ", "ʂ", "ʐ", "ts", "tsh", "s"],
        "perceived_s": ["k", "kh"],
        "number": 0
    }

    Fricative = {
        "canonical_s": ["p", "ph", "m", "t", "th", "n", "l", "k", "kh", "x", "tɕ", "thɕ", "tʂ", "thʂ", "ts", "tsh"],
        "perceived_s": ["f", "ʂ", "s", "ɕ", "ʐ"],
        "number": 0
    }

    Affricate = {
        "canonical_s": ["p", "ph", "m", "f", "t", "th", "n", "l", "k", "kh", "x",  "ɕ",  "ʂ", "ʐ", "s"],
        "perceived_s": ["tɕ", "thɕ", "tʂ", "thʂ", "ts", "tsh"],
        "number": 0
    }

    FCDP = {
        "canonical_s": ["an", "ən", "aŋ", "əŋ"],
        "perceived_s": ["a"],
        "canonical_d": ["an", "ən", "aŋ", "əŋ"],
        "perceived_d": ["<eps>"],
        "number": 0
    }
    consonants = ["p", "ph", "m", "f", "t", "th", "n", "l", "x", "tɕ",
                  "thɕ", "ɕ", "tʂ", "thʂ", "ʂ", "ʐ", "ts", "tsh", "s", "k", "kh"]
    consonant_error_types = ["Stopping", "Backing", "Fricative", "Affricate"]
    phoneme_unit = {
        "t": "ㄉ",
        "a": "ㄚ",
        "kh": "ㄎ",
        "u": "ㄨ",
        "k": "ㄍ",
        "i": "ㄧ",
        "m": "ㄇ",
        "tɕ": "ㄐ",
        "p": "ㄅ",
        "x": "ㄏ",
        "l": "ㄌ",
        "o": "ㄛ",
        "n": "ㄋ",
        "ei": "ㄟ",
        "ou": "ㄡ",
        "f": "ㄈ",
        "ph": "ㄆ",
        "ts": "ㄗ",
        "thɕ": "ㄑ",
        "tsh": "ㄘ",
        "au": "ㄠ",
        "an": "ㄢ",
        "aŋ": "ㄤ",
        "s": "ㄙ",
        "əŋ": "ㄥ",
        "ən": "ㄣ",
        "ʐ": "ㄖ",
        "th": "ㄊ",
        "ɤ": "ㄜ",
        "ɕ": "ㄒ",
        "y": "ㄩ",
        "ai": "ㄞ",
        "e": "ㄝ"
    }
    error_dic = {}
    error_number_dic = {}
    error_number_zhuyin_dic = {}
    total_consonant = 0
    total_vowel = 0
    PCC_error = 0
    hyps = []
    others = 0

    for uterranceID in dic.keys():
        data = dic[uterranceID]
        ref = dic[uterranceID][0].split(" ")
        hyp = dic[uterranceID][1].split(" ")
        OP = dic[uterranceID][2].split(" ")
        # 這邊是看有沒有要看辨識結果(注音)
        tmp = " ".join([phoneme_unit[n] for n in hyp if n != '<eps>'])
        hyps.append("_".join(uterranceID.split("_")[1:])+": "+tmp)
        for i in range(len(OP)):
            exist_in_error_type = False
            if (ref[i] in consonants):
                total_consonant = total_consonant + 1
                for consonant_error_type in consonant_error_types:
                    error_type_dic = eval(consonant_error_type)
                    if (ref[i] in error_type_dic["canonical_s"] and hyp[i] in error_type_dic["perceived_s"]):
                        error_type_dic["number"] += 1
                        exist_in_error_type = True
                if (~exist_in_error_type and OP[i] != 'C'):
                    others += 1
            elif (ref[i] != '<eps>'):
                total_vowel += 1
                if (ref[i] in FCDP["canonical_s"] and hyp[i] in FCDP["perceived_s"]):
                    FCDP["number"] += 1
                    exist_in_error_type = True
                if (ref[i] in FCDP["canonical_d"] and hyp[i] in FCDP["perceived_d"]):
                    FCDP["number"] += 1
                    exist_in_error_type = True
                if (~exist_in_error_type and OP[i] != 'C'):
                    others += 1
    re_dict = {}
    chinese_dict = {
        "Stopping": "塞音化",
        "Backing": "舌根音化",
        "Fricative": "擦音化",
        "Affricate": "塞擦音化"
    }
    for consonant_error_type in consonant_error_types:
        error_type_dic = eval(consonant_error_type)
        print(f'error_type_dic: {error_type_dic}')

        pen = round(error_type_dic["number"]/total_consonant, 2)
        PCC_error += error_type_dic["number"]
        print(f"{consonant_error_type} : {pen}")
        re_dict[chinese_dict[consonant_error_type]] = pen

    FCDP_number = FCDP["number"]
    print(f"FCDP :{FCDP_number/total_vowel}")
    print(f"total_consonant:{total_consonant}")
    re_dict["聲隨韻母"] = round(FCDP_number / total_vowel, 2)
    re_dict["子音正確率"] = round(1 - PCC_error / total_consonant, 2)
    others_ratio = others / (total_consonant + total_vowel)
    re_dict["其他"] = others_ratio

    print(f"PCC : {1 - PCC_error/total_consonant}")
    
    hyps.sort()
    # 看有沒有要看結果
    # for hyp in hyps:
    #     re_str = re_str+hyp+'\n'
    # Convert the dictionary to a JSON-formatted string
    re_str = json.dumps(re_dict, ensure_ascii=False, indent=4)
    return re_str


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)