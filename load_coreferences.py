
import re 

def get_gold_data(doc):
    GOLD_DATA_FILE = "/home/flavio/projects/rel20/data/generic/test_datasets/AIDA/AIDA-YAGO2-dataset.tsv"
    entities = []

    in_file = open(GOLD_DATA_FILE, "r")
    for line in in_file:
        if re.search(f"^-DOCSTART- \({doc} ", line):
            break
    for line in in_file:
        if re.search(f"^-DOCSTART- ", line):
            break
        fields = line.strip().split("\t")
        if len(fields) > 3:
            if fields[1] == "B":
                entities.append([fields[2], fields[3]])
    return entities


def find_coref(mention, mentlist, verbose=False):
    "re-implement __find_coref from REL"

    pairs = []
    cur_m = mention[0].lower() 
    cur_m_entity = mention[1] # entity of the current mention
    for m in mentlist:
        entity = m[1]
        m = m[0].lower()
        if cur_m == m:
            continue 
        start_pos = m.find(cur_m)
        if start_pos == -1:
            continue 
        end_pos = start_pos + len(cur_m) - 1

        if (entity == cur_m_entity) and (start_pos == 0 or m[start_pos - 1] == " "):
            if end_pos == len(m) - 1:
                if verbose:
                    print(f"{cur_m} is a coref for {m} ")
                pair = (m, cur_m)
                pairs.append(pair)
            elif m[end_pos + 1] == " ":
                if verbose:
                    print(f"{cur_m} is a coref for {m} ")
                pair = (m, cur_m)
                pairs.append(pair)
            else:
                continue
    return pairs 


aida_docnames = [
    '1163testb', '1164testb', '1165testb', '1168testb', '1172testb', '1173testb', '1175testb', 
    '1177testb', '1178testb', '1182testb', '1184testb', '1186testb', '1188testb', '1189testb', 
    '1190testb', '1193testb', '1194testb', '1201testb', '1202testb', '1204testb', '1206testb', 
    '1208testb', '1213testb', '1215testb', '1216testb', '1217testb', '1223testb', '1224testb', 
    '1238testb', '1241testb', '1245testb', '1251testb', '1258testb', '1259testb', '1264testb', 
    '1267testb', '1268testb', '1270testb', '1271testb', '1275testb', '1279testb', '1280testb', 
    '1288testb', '1299testb', '1303testb', '1310testb', '1312testb', '1315testb', '1326testb',
    '1331testb']


def load_pairs():
    "load ground truth coreference pairs"
    coref_pairs = {}

    for doc in aida_docnames:
        gold_entities = get_gold_data(doc)
        pairs = []
        for m in gold_entities:
            p = find_coref(m, gold_entities)
            if p != []:
                for pair in p:
                    pairs.append(pair)

        coref_pairs[doc] = pairs
    
    return coref_pairs


def load_coreferences():
    "transform ground truth coref pairs to list of mentions"

    coref_pairs = load_pairs()
    mentions = set()
    for pairs in coref_pairs.values():
        for p in pairs:
            mentions.add(p[0])
            mentions.add(p[1])
    
    return mentions


