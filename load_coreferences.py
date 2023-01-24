
import re 

docnames = {
    "aida_testb": [
        '1163testb', '1164testb', '1165testb', '1168testb', '1172testb', '1173testb', '1175testb', 
        '1177testb', '1178testb', '1182testb', '1184testb', '1186testb', '1188testb', '1189testb', 
        '1190testb', '1193testb', '1194testb', '1201testb', '1202testb', '1204testb', '1206testb', 
        '1208testb', '1213testb', '1215testb', '1216testb', '1217testb', '1223testb', '1224testb', 
        '1238testb', '1241testb', '1245testb', '1251testb', '1258testb', '1259testb', '1264testb', 
        '1267testb', '1268testb', '1270testb', '1271testb', '1275testb', '1279testb', '1280testb', 
        '1288testb', '1299testb', '1303testb', '1310testb', '1312testb', '1315testb', '1326testb',
        '1331testb'
    ],
    "aida_testa": [
        '947testa', '948testa', '949testa', '950testa', '951testa', '952testa', '953testa', '954testa', '955testa', 
        '956testa', '957testa', '958testa', '959testa', '960testa', '961testa', '962testa', '963testa', '964testa',
        '965testa', '966testa', '967testa', '968testa', '969testa', '970testa', '971testa', '972testa', '973testa', 
        '974testa', '975testa', '976testa', '977testa', '978testa', '979testa', '980testa', '981testa', '982testa', 
        '983testa', '984testa', '985testa', '986testa', '987testa', '988testa', '989testa', '990testa', '991testa', 
        '992testa', '993testa', '994testa', '995testa', '996testa', '997testa', '998testa', '999testa', '1000testa', 
        '1001testa', '1002testa', '1003testa', '1004testa', '1005testa', '1006testa', '1007testa', '1008testa', '1009testa',
        '1010testa', '1011testa', '1012testa', '1013testa', '1014testa', '1015testa', '1016testa', '1017testa', '1018testa',
        '1019testa', '1020testa', '1021testa', '1022testa', '1023testa', '1024testa', '1025testa', '1026testa', '1027testa',
        '1028testa', '1029testa', '1030testa', '1031testa', '1032testa', '1033testa', '1034testa', '1035testa', '1036testa',
        '1037testa', '1038testa', '1039testa', '1040testa', '1041testa', '1042testa', '1043testa', '1044testa', '1045testa', 
        '1046testa', '1047testa', '1048testa', '1049testa', '1050testa', '1051testa', '1052testa', '1053testa', '1054testa', 
        '1055testa', '1056testa', '1057testa', '1058testa', '1059testa', '1060testa', '1061testa', '1062testa', '1063testa',
        '1064testa', '1065testa', '1066testa', '1067testa', '1068testa', '1069testa', '1070testa', '1071testa', '1072testa',
        '1073testa', '1074testa', '1075testa', '1076testa', '1077testa', '1078testa', '1079testa', '1080testa', '1081testa',
        '1082testa', '1083testa', '1084testa', '1085testa', '1086testa', '1087testa', '1088testa', '1089testa', '1090testa',
        '1091testa', '1092testa', '1093testa', '1094testa', '1095testa', '1096testa', '1097testa', '1098testa', '1099testa', 
        '1100testa', '1101testa', '1102testa', '1103testa', '1104testa', '1105testa', '1106testa', '1107testa', '1108testa',
        '1109testa', '1110testa', '1111testa', '1112testa', '1113testa', '1114testa', '1115testa', '1116testa', '1117testa',
        '1118testa', '1119testa', '1120testa', '1121testa', '1122testa', '1123testa', '1124testa', '1125testa', '1126testa',
        '1127testa', '1128testa', '1129testa', '1130testa', '1131testa', '1132testa', '1133testa', '1134testa', '1135testa',
        '1136testa', '1137testa', '1138testa', '1139testa', '1140testa', '1141testa', '1142testa', '1143testa', '1144testa',
        '1145testa', '1146testa', '1147testa', '1148testa', '1149testa', '1150testa', '1151testa', '1152testa', '1153testa',
        '1154testa', '1155testa', '1156testa', '1157testa', '1158testa', '1159testa', '1160testa', '1161testa', '1162testa'
    ]
}


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



def load_pairs(validate_or_test="test"):
    """
    load ground truth coreference pairs.

    Parameters:
    ----------
    validate_or_test: should validation data (aida_testA) or test data (aida_testB) be used?
    """
    assert validate_or_test in ["test", "validate"]
    coref_pairs = {}
    docname_type = "aida_testb"
    if validate_or_test == "validate":
        docname_type = "aida_testa"

    for doc in docnames[docname_type]:
        gold_entities = get_gold_data(doc)
        pairs = []
        for m in gold_entities:
            p = find_coref(m, gold_entities)
            if p != []:
                for pair in p:
                    pairs.append(pair)

        coref_pairs[doc] = pairs
    
    return coref_pairs


def load_coreferences(drop_duplicates=True, validate_or_test="test"):
    "transform ground truth coref pairs to list of mentions"

    coref_pairs = load_pairs(validate_or_test=validate_or_test)
    if drop_duplicates:
        mentions = set()
        for pairs in coref_pairs.values():
            for p in pairs:
                mentions.add(p[0])
                mentions.add(p[1])
    else:
        mentions = []
        for pairs in coref_pairs.values():
            for p in pairs:
                mentions.append(p[0])
                mentions.append(p[1])
    
    return mentions


