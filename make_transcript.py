

# load json module
import json

# python dictionary with key value pairs
transcript_dict={
    "wordcard03_01": "sil t a sil kh u sil",
    "wordcard03_01_1": "sil t a sil",
    "wordcard03_01_2": "sil kh u sil",
    "wordcard03_02": "sil t a sil k u sil",
    "wordcard03_02_1": "sil t a sil",
    "wordcard03_02_2": "sil k u sil",
    "wordcard03_03": "sil tɕ i sil m u sil",
    "wordcard03_03_1": "sil tɕ i sil",
    "wordcard03_03_2": "sil m u sil",
    "wordcard03_04": "sil th a sil p u sil",
    "wordcard03_04_1": "sil th a sil",
    "wordcard03_04_2": "sil p u sil",
    "wordcard03_05": "sil tɕ i sil th a sil",
    "wordcard03_05_1": "sil tɕ i sil",
    "wordcard03_05_2": "sil th a sil",
    "wordcard03_06": "sil x a sil l u o sil",
    "wordcard03_06_1": "sil x a sil",
    "wordcard03_06_2": "sil l u o sil",
    "wordcard03_07": "sil n ei sil kh u sil",
    "wordcard03_07_1": "sil n ei sil",
    "wordcard03_07_2": "sil kh u sil",
    "wordcard03_08": "sil k u sil th ou sil",
    "wordcard03_08_1": "sil k u sil",
    "wordcard03_08_2": "sil th ou sil",
    "wordcard04_01": "sil th ou sil f a sil",
    "wordcard04_01_1": "sil th ou sil",
    "wordcard04_01_2": "sil f a sil",
    "wordcard04_02": "sil ph i sil ph a sil",
    "wordcard04_02_1": "sil ph i sil",
    "wordcard04_02_2": "sil ph a sil",
    "wordcard04_03": "sil thɕ i sil ts sil",
    "wordcard04_03_1": "sil thɕ i sil",
    "wordcard04_03_2": "sil ts sil",
    "wordcard04_04": "sil thɕ i sil f u sil",
    "wordcard04_04_1": "sil thɕ i sil",
    "wordcard04_04_2": "sil f u sil",
    "wordcard04_05": "sil tsh a sil i au sil",
    "wordcard04_05_1": "sil tsh a sil",
    "wordcard04_05_2": "sil i au sil",
    "wordcard04_06": "sil tsh au sil m ei sil",
    "wordcard04_06_1": "sil tsh au sil",
    "wordcard04_06_2": "sil m ei sil",
    "wordcard04_07": "sil ts ou sil l u sil",
    "wordcard04_07_1": "sil ts ou sil",
    "wordcard04_07_2": "sil l u sil",
    "wordcard04fcdp_01": "sil s an sil s aŋ sil",
    "wordcard04fcdp_01_1": "sil s an sil",
    "wordcard04fcdp_01_2": "sil s aŋ sil",
    "wordcard04fcdp_02": "sil m i an sil i aŋ sil",
    "wordcard04fcdp_02_1": "sil m i an sil",
    "wordcard04fcdp_02_2": "sil i aŋ sil",
    "wordcard04fcdp_03": "sil s əŋ sil i ən sil",
    "wordcard04fcdp_03_1": "sil s əŋ sil",
    "wordcard04fcdp_03_2": "sil i ən sil",
    "wordcard04fcdp_04": "sil ɕ i əŋ sil ɕ i əŋ sil",
    "wordcard04fcdp_04_1": "sil ɕ i əŋ sil",
    "wordcard04fcdp_04_2": "sil ɕ i əŋ sil",
    "wordcard05_01": "sil ʐ sil l i sil",
    "wordcard05_01_1": "sil ʐ sil",
    "wordcard05_01_2": "sil l i sil",
    "wordcard05_02": "sil ts sil ts u sil",
    "wordcard05_02_1": "sil ts sil",
    "wordcard05_02_2": "sil ts u sil",
    "wordcard05_03": "sil ts u sil s u sil",
    "wordcard05_03_1": "sil ts u sil",
    "wordcard05_03_2": "sil s u sil",
    "wordcard05_04": "sil ɕ i sil s ou sil",
    "wordcard05_04_1": "sil ɕ i sil",
    "wordcard05_04_2": "sil s ou sil",
    "wordcard05_05": "sil tsh ou sil th i sil",
    "wordcard05_05_1": "sil tsh ou sil",
    "wordcard05_05_2": "sil th i sil",
    "wordcard05_06": "sil ɕ i a sil tsh ɤ sil",
    "wordcard05_06_1": "sil ɕ i a sil",
    "wordcard05_06_2": "sil tsh ɤ sil",
    "wordcard05_07": "sil s u sil ts sil",
    "wordcard05_07_1": "sil s u sil",
    "wordcard05_07_2": "sil ts sil",
    "wordcard05_08": "sil th u sil s sil",
    "wordcard05_08_1": "sil th u sil",
    "wordcard05_08_2": "sil s sil"
}

# create json object from dictionary
write_json = json.dumps(transcript_dict)

# open file for writing, "w" 
f = open("transcript_dict.json","w")

# write json object to file
f.write(write_json)

# close file
f.close()


with open("transcript_dict.json") as f:
    dict = json.loads(f.read())
print(type(dict))
print(dict)