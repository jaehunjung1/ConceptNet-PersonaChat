import re
import ipdb
import json



def preprocess(chunk, idx):
    obj = {"id": idx, "dialog": [], "user_profile": [], "bot_profile": []}

    for row in chunk:
        row = re.sub(r'^\d+ ', '', row)

        if row.startswith("your persona: "):
            obj["user_profile"].append(re.sub(r"^your persona: ", "", row).strip("\n"))
        elif row.startswith("partner's persona: "):
            obj["bot_profile"].append(re.sub(r"^partner's persona: ", "", row).strip("\n"))
        else:
            tmp = list(map(lambda x: x.strip("\n"), row.split("\t")))
            obj["dialog"].extend([{"text": tmp[0]}, {"text": tmp[1]}])

    return obj


if __name__ == "__main__":
    with open("./persona_data/train_both_original_no_cands.txt", "r") as f:
        arr = []
        while True:
            line = f.readline()
            if not line:
                break
            else:
                arr.append(line)

    json_arr = []
    chunk = []
    idx = 0
    for row in arr:
        if row.startswith("1 "):
            if len(chunk) > 0:
                json_arr.append(preprocess(chunk, idx))
                idx += 1
            chunk = [row]

        else:
            chunk.append(row)

    with open('./persona_data/train_both_original_no_cands.json', 'w') as out:
        json.dump(json_arr, out)
