import pandas as pd
import numpy as np
import os

data_path = ""
Antibod_updated = pd.read_csv(os.path.join("./data/" ,'ADA_summary_version_1.csv'), low_memory= False , sep= ',' )
Antibod_train   = pd.read_csv(os.path.join("../AntiBERTy-main/data/", 'train_2%.csv'), low_memory=False, sep=',')
Antibod_test    = pd.read_csv(os.path.join("../AntiBERTy-main/data/", 'test_2%.csv'), low_memory=False, sep=',')


Antibod_updated.rename(columns={'Antibody drug name': 'Therapeutic_antibody'}, inplace=True)

Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'human', 'antibody_type'] = 0
Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'humanized', 'antibody_type'] = 0
Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'chimeric', 'antibody_type'] = 0
Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'mouse', 'antibody_type'] = 1


Antibod_updated.set_index('Therapeutic_antibody')['Type of antibodies']
###
Antibod_old = pd.concat([Antibod_train,Antibod_test])

Antibod_old["Therapeutic_antibody"] = Antibod_old["Therapeutic_antibody"].str.strip().str.lower()
Antibod_updated["Therapeutic_antibody"] = Antibod_updated["Therapeutic_antibody"].str.strip().str.lower()

#merged_df = pd.merge(Antibod_old, Antibod_updated, on='Therapeutic_antibody', how='inner')
#merged_df.to_csv('shared_dataset.csv')

# Names only in Antibod_old
only_in_Antibod_old = Antibod_old[~Antibod_old['Therapeutic_antibody'].isin(Antibod_updated['Therapeutic_antibody'])]
print("only in Wang's Antibod")
print(only_in_Antibod_old['Therapeutic_antibody'])
print("##############################################################################################################################")
# Names only in Antibod_updated
only_in_Antibod_updated = Antibod_updated[~Antibod_updated['Therapeutic_antibody'].isin(Antibod_old['Therapeutic_antibody'])]
print("only in Ji Young's Antibod")
print(only_in_Antibod_updated['Therapeutic_antibody'])
print("##############################################################################################################################")

threshold = 2.0

print("Generating test Dataset")
test = pd.DataFrame()
test["N0"] =  Antibod_test["N0"]
test['Therapeutic_antibody'] = Antibod_test["Therapeutic_antibody"]
test['Heavy_Chain'] = Antibod_test["Heavy_Chain"]
test['Light_Chain'] = Antibod_test["Light_Chain"]
test["Therapeutic_antibody"] = test["Therapeutic_antibody"].str.strip().str.lower()
Immun_value_map = Antibod_updated.set_index('Therapeutic_antibody')['ADA (%) we found (study reports & FDA labels)']
Immun_type_map = Antibod_updated.set_index('Therapeutic_antibody')['antibody_type']
test['Immun_value'] = test['Therapeutic_antibody'].map(Immun_value_map)
test['antibody_type'] = test['Therapeutic_antibody'].map(Immun_type_map)
test['Immun_label'] = np.where(test['Immun_value'] < threshold, 0, 1)
#test['Immun_label'] = np.where(test['Immun_value'] < 2, 0,    np.where(test['Immun_value'] <= 4, 1, 2))
test = test.dropna(axis=0)

test.to_csv('data/test_ver1_' + str(threshold) + '%.csv', index=False)
print("##############################################################################################################################")

print("Generating Train Dataset")
train = pd.DataFrame()
train["N0"] = Antibod_train["N0"]
train['Therapeutic_antibody'] = Antibod_train["Therapeutic_antibody"]
train["Therapeutic_antibody"] = train["Therapeutic_antibody"].str.strip().str.lower()
if (only_in_Antibod_updated.shape[0] > 0):    train = pd.concat([train, only_in_Antibod_updated[['Therapeutic_antibody']]], axis=0)
train = train.reset_index(drop=True)
train['N0'] = train.index
heavy_map = Antibod_old.set_index('Therapeutic_antibody')['Heavy_Chain']
train['Heavy_Chain'] = train['Therapeutic_antibody'].map(heavy_map)
light_map = Antibod_old.set_index('Therapeutic_antibody')['Light_Chain']
train['Light_Chain'] = train['Therapeutic_antibody'].map(light_map)
Immun_value_map = Antibod_updated.set_index('Therapeutic_antibody')['ADA (%) we found (study reports & FDA labels)']
Immun_type_map = Antibod_updated.set_index('Therapeutic_antibody')['antibody_type']
train['Immun_value'] = train['Therapeutic_antibody'].map(Immun_value_map)
train['antibody_type'] = train['Therapeutic_antibody'].map(Immun_type_map)
train['Immun_label'] = np.where(train['Immun_value'] < threshold, 0, 1)
#train['Immun_label'] = np.where(train['Immun_value'] < 2, 0,    np.where(train['Immun_value'] <= 4, 1, 2))
train.loc[train['Therapeutic_antibody'] == 'tibulizumab', 'Heavy_Chain'] = "QVQLVQSGAEVKKPGSSVKVSCKASGYSFTDYHIHWVRQAPGQGLEWMGVINPMYGTTDYNQRFKGRVTITADESTSTAYMELSSLRSEDTAVYYCARYDYFTGTGVYWGQGTLVTVSS"
train.loc[train['Therapeutic_antibody'] == 'tibulizumab', 'Light_Chain'] = "DIVMTQTPLSLSVTPGQPASISCRSSRSLVHSRGNTYLHWYLQKPGQSPQLLIYKVSNRFIGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCSQSTHLPFTFGQGTKLEIK"

train.loc[train['Therapeutic_antibody'] == 'pinatuzumab', 'Heavy_Chain'] = "EVQLVESGGGLVQPGGSLRLSCAASGYEFSRSWMNWVRQAPGKGLEWVGRIYPGDGDTQYSGKFKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARDGSSWDWYFDVWGQGTLVTVSS"
train.loc[train['Therapeutic_antibody'] == 'pinatuzumab', 'Light_Chain'] = "DIQMTQSPSSLSASVGDRVTITCRSSQSIVHSVGNTFLEWYQQKPGKAPKLLIYKVSNRFSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCFQGSQFPYTFGQGTKVEIK"


train = train.dropna(axis=0)
train.to_csv('data/train_ver1_' + str(threshold) + '%.csv', index=False)

