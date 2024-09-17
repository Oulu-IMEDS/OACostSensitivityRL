import hydra
import pandas as pd
import glob
import os
from tqdm import tqdm
from sas7bdat import SAS7BDAT
import numpy as np
import matplotlib.pyplot as plt
from data.parse_metadata import ParseMetaDataOAI

class GenerateDataFrameOAI(ParseMetaDataOAI):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.cfg = cfg
        self.path = os.path.join(cfg.root_path, cfg.metadata_path)
        self.save_path = os.path.join(cfg.root_path, cfg.dataframe_path)

    def JSW_progression(self):
        mode = 'landscape'
        data_filename = f'JSW_ClinicalVars_Progression_verK'
        if not os.path.exists(os.path.join(self.save_path, f'{data_filename}.csv')):
            df_JSW = self.parsing_JSW(mode=mode)
            cols = ['ID', 'SIDE']
            cols.extend(col for col in df_JSW.columns if 'MJSW250' in col)
            cols.append('minJSW@0')
            df_JSW_small = df_JSW[cols]
            df_clinical = self.merge_clinical_tables()

            data = []
            for i in range(len(df_JSW)):
                row_dict = {}
                jsw_baseline = df_JSW.iloc[i]['MJSW250@0']
                row_dict['ID'] = df_JSW.iloc[i]['ID']
                row_dict['SIDE'] = df_JSW.iloc[i]['SIDE']
                minJSW_baseline = df_JSW.iloc[i][f'minJSW@0']
                #     print(len(row_dict['Progress']))
                for j in range(1, self.follow_ups_len):
                    visit = self.encode_visit[self.follow_ups[j]]
                    jsw_followup = df_JSW.iloc[i][f'MJSW250@{visit}']

                    if np.isnan(jsw_followup):
                        continue

                    if jsw_baseline - jsw_followup >= 0.7:
                        row_dict['Progress'] = np.int(j)
                        break
                    elif jsw_baseline - jsw_followup < 0:
                        jsw_baseline = jsw_followup
                    else:
                        #   no progressed
                        row_dict['Progress'] = 0


                data.append(row_dict)
            df_progress = pd.DataFrame(data=data)
            df_progress.dropna(subset=['Progress'], inplace=True)


            df_JSW_with_progressed = pd.merge(df_JSW_small, df_progress, on=['ID', 'SIDE'], how='inner')
            full_df = pd.merge(df_JSW_with_progressed, df_clinical, on=['ID', 'SIDE'], how='left')
            full_df = full_df.replace({'SIDE': {'L': 0, 'R': 1}})
            full_df = full_df.drop_duplicates(subset=['ID', 'SIDE'])
            full_df.to_csv(os.path.join(self.save_path, f'{data_filename}.csv'), index=False)
            full_df.to_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'), protocol=4)
        else:
            full_df = pd.read_csv(os.path.join(self.save_path, f'{data_filename}.csv'))

        return full_df

    def multi_JSW_progression(self):
        mode = 'landscape'
        data_filename = f'JSW_ClinicalVars_Multi_Progression_verK'
        if not os.path.exists(os.path.join(self.save_path, f'{data_filename}.pkl')):
            df_JSW = self.parsing_JSW(mode=mode)
            cols = ['ID', 'SIDE']
            cols.extend(col for col in df_JSW.columns if 'MJSW250' in col)
            cols.append('minJSW@0')
            df_JSW_small = df_JSW[cols]
            df_clinical = self.merge_clinical_tables()

            data = []
            for i in range(len(df_JSW)):
                row_dict = {}
                jsw_baseline = df_JSW.iloc[i]['MJSW250@0']
                row_dict['ID'] = df_JSW.iloc[i]['ID']
                row_dict['SIDE'] = df_JSW.iloc[i]['SIDE']
                row_dict['Progress'] = []
                # if jsw_baseline < 1.6:
                #     row_dict['Progress'] = np.nan
                #     continue
                #     print(len(row_dict['Progress']))
                for j in range(1, self.follow_ups_len):
                    visit = self.encode_visit[self.follow_ups[j]]
                    jsw_followup = df_JSW.iloc[i][f'MJSW250@{visit}']

                    if np.isnan(jsw_followup):
                        continue

                    if jsw_baseline - jsw_followup >= 0.7:
                        row_dict['Progress'].append(np.int(j))
                        jsw_baseline = jsw_followup
                    # elif jsw_baseline - jsw_followup < 0:
                        # jsw_baseline = jsw_followup
                    else:
                        #   no progressed
                        continue

                data.append(row_dict)
            df_progress = pd.DataFrame(data=data)
            df_progress.dropna(subset=['Progress'], inplace=True)

            df_JSW_with_progressed = pd.merge(df_JSW_small, df_progress, on=['ID', 'SIDE'], how='inner')
            full_df = pd.merge(df_JSW_with_progressed, df_clinical, on=['ID', 'SIDE'], how='left')
            full_df = full_df.replace({'SIDE': {'L': 0, 'R': 1}})
            full_df = full_df.drop_duplicates(subset=['ID', 'SIDE'])
            full_df.to_csv(os.path.join(self.save_path, f'{data_filename}.csv'), index=False)
            full_df.to_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'), protocol=4)
        else:
            full_df = pd.read_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'))

        return full_df

    def KL_progression(self, data_filename=None):
        mode = 'landscape'
        df_KL = self.parsing_KL_grade(mode=mode)
        # df_KL.replace({0: 1}, inplace=True)
        df_clinical = self.merge_clinical_tables()

        data = []
        for i in range(len(df_KL)):
            row_dict = {}
            kl_baseline = df_KL.iloc[i]['KL@0']
            row_dict['ID'] = df_KL.iloc[i]['ID']
            row_dict['SIDE'] = df_KL.iloc[i]['SIDE']
            #     print(len(row_dict['Progress']))
            for j in range(1, self.follow_ups_len):
                visit = self.encode_visit[self.follow_ups[j]]
                kl_followup = df_KL.iloc[i][f'KL@{visit}']
                # if kl_baseline == 4:
                #     row_dict['Progress'] = -1
                #     break

                if np.isnan(kl_followup):
                    continue

                if kl_followup - kl_baseline >= 1:
                    #   first progressed
                    row_dict['Progress'] = np.int(j)
                    break
                elif kl_followup - kl_baseline < 0:
                    #   first progressed
                    row_dict['Progress'] = -1
                    break
                else:
                    #   no progressed
                    row_dict['Progress'] = 0

            data.append(row_dict)
        df_progress = pd.DataFrame(data=data)
        df_progress.dropna(subset=['Progress'], inplace=True)


        df_KL_with_progressed = pd.merge(df_KL, df_progress, on=['ID', 'SIDE'], how='inner')
        full_df = pd.merge(df_KL_with_progressed, df_clinical, on=['ID', 'SIDE'], how='left')
        full_df = full_df.replace({'SIDE': {'L': 0, 'R': 1}})
        full_df = full_df.drop_duplicates(subset=['ID', 'SIDE'])
        if data_filename is not None:
            full_df.to_csv(os.path.join(self.save_path, f'{data_filename}.csv'), index=False)
            full_df.to_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'), protocol=4)

        return full_df

    def multi_KL_progression(self):
        mode = 'landscape'
        data_filename = f'KL_ClinicalVars_Multi_Progression_verK'
        if not os.path.exists(os.path.join(self.save_path, f'{data_filename}.cs')):
            df_KL = self.parsing_KL_grade(mode=mode)
            # df_KL.replace({0: 1}, inplace=True)
            df_clinical = self.merge_clinical_tables()

            data = []
            for i in range(len(df_KL)):
                row_dict = {}
                kl_baseline = df_KL.iloc[i]['KL@0']
                row_dict['ID'] = df_KL.iloc[i]['ID']
                row_dict['SIDE'] = df_KL.iloc[i]['SIDE']
                row_dict['Progress'] = []
                for j in range(1, self.follow_ups_len):
                    visit = self.encode_visit[self.follow_ups[j]]
                    kl_followup = df_KL.iloc[i][f'KL@{visit}']
                    # if kl_baseline == 4:
                    #     row_dict['Progress'] = -1
                    #     break

                    if np.isnan(kl_followup):
                        continue

                    if kl_followup - kl_baseline >= 1:
                        #   first progressed
                        row_dict['Progress'].append(np.int(j))
                        kl_baseline = kl_followup
                    elif kl_followup - kl_baseline < 0:
                        #   improvement
                        # row_dict['Progress'] = np.nan
                        pass
                    else:
                        #   no progressed
                        pass
                        # row_dict['Progress'] = 0

                data.append(row_dict)
            df_progress = pd.DataFrame(data=data)
            df_progress.dropna(subset=['Progress'], inplace=True)

            df_KL_with_progressed = pd.merge(df_KL, df_progress, on=['ID', 'SIDE'], how='inner')
            full_df = pd.merge(df_KL_with_progressed, df_clinical, on=['ID', 'SIDE'], how='left')
            full_df = full_df.replace({'SIDE': {'L': 0, 'R': 1}})
            full_df = full_df.drop_duplicates(subset=['ID', 'SIDE'])
            full_df.to_csv(os.path.join(self.save_path, f'{data_filename}.csv'), index=False)
            full_df.to_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'), protocol=4)
        else:
            full_df = pd.read_csv(os.path.join(self.save_path, f'{data_filename}.csv'))

        return full_df

    def SF12_progression(self):
        mode = 'landscape'
        data_filename = f'SF12_ClinicalVars_Progression_verP'
        df_KL = self.parsing_KL_grade(mode=mode)
        df_KL = df_KL.replace({'SIDE': {'L': 0, 'R': 1}})
        # df_KL_progression = self.KL_progression()
        # kl_features_headers = ['ID', 'SIDE','Progress']
        # kl_features_headers.extend([col for col in df_KL_progression.columns if f'KL@' in col])
        # df_KL_progression = df_KL_progression[kl_features_headers]
        # df_KL_progression.rename(columns={'Progress':'Progress_KL'}, inplace=True)
        if not os.path.exists(os.path.join(self.save_path, f'{data_filename}.csv')):
            df_clinical = self.merge_clinical_tables()
            features_headers = ['ID','SIDE']
            features_headers.extend([col for col in df_clinical.columns if f'SF12@' in col])
            df_sf12 = df_clinical[features_headers]
            data = []
            for i in range(len(df_sf12)):
                row_dict = {}
                sf12_baseline = df_sf12.iloc[i]['SF12@0']
                row_dict['ID'] = df_sf12.iloc[i]['ID']
                row_dict['SIDE'] = df_sf12.iloc[i]['SIDE']
                #     print(len(row_dict['Progress']))
                for j in range(1, self.follow_ups_len):
                    visit = self.encode_visit[self.follow_ups[j]]
                    sf12_followup = df_sf12.iloc[i][f'SF12@{visit}']

                    if np.isnan(sf12_followup):
                        continue

                    if sf12_baseline - sf12_followup >= 7.0:
                        #   first progressed
                        row_dict['Progress'] = np.int(j)
                        break
                    else:
                        #   no progressed
                        row_dict['Progress'] = 0
                        sf12_baseline = sf12_followup

                data.append(row_dict)
            df_progress = pd.DataFrame(data=data)
            df_progress.dropna(subset=['Progress'], inplace=True)


            df_SF12_with_progressed = pd.merge(df_progress, df_clinical, on=['ID','SIDE'], how='left')
            df_SF12_with_progressed  = df_SF12_with_progressed.replace({'SIDE': {'L': 0, 'R': 1}})
            full_df = pd.merge(df_SF12_with_progressed, df_KL, on=['ID','SIDE'], how='left')
            # full_df = full_df.replace({'SIDE': {'L': 0, 'R': 1}})
            full_df = full_df.drop_duplicates(subset=['ID','SIDE'])
            full_df.to_csv(os.path.join(self.save_path, f'{data_filename}.csv'), index=False)
            full_df.to_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'), protocol=4)
        else:
            full_df = pd.read_csv(os.path.join(self.save_path, f'{data_filename}.csv'))

        return full_df

    def merge_clinical_tables(self):
        mode = 'landscape'
        if not os.path.exists(os.path.join(self.save_path, f'Clinical_vars_landscape.csv')):
            df_patient = self.parsing_patient_info(mode)
            df_koos = self.parsing_KOOSQoL_score(mode)
            df_patient_koos = pd.merge(df_patient, df_koos, on=['ID'], how='inner')
            df_pas = self.parsing_PA_score(mode)
            df_patient_koos_pas = pd.merge(df_patient_koos, df_pas, on=['ID'], how='inner')
            df_sf12 = self.parsing_SF_12_score(mode)
            df_patient_koos_pas_sf = pd.merge(df_patient_koos_pas, df_sf12, on=['ID'], how='inner')
            df_womac = self.parsing_WOMAC_score(mode)
            df_patient_koos_pas_sf_womac = pd.merge(df_patient_koos_pas_sf, df_womac, on=['ID'], how='inner')
            df_clinical_vars = self.parsing_other_clinical_variables(mode)
            df_clinical_merged_final = pd.merge(df_patient_koos_pas_sf_womac, df_clinical_vars, on=['ID', 'SIDE'], how='inner')
            cols = df_clinical_merged_final.columns.tolist()
            cols.remove('SIDE')
            cols.insert(1,'SIDE')
            df_clinical_merged_final = df_clinical_merged_final[cols]
            df_clinical_merged_final.to_csv(os.path.join(self.save_path, f'Clinical_vars_landscape.csv'), index=False)
            df_clinical_merged_final.to_pickle(os.path.join(self.save_path, f'Clinical_vars_landscape.pkl'), protocol=4)
        else:
            df_clinical_merged_final = pd.read_csv(os.path.join(self.save_path, f'Clinical_vars_landscape.csv'))

        return df_clinical_merged_final

    def KL_JSW_merge_dataframe(self, data_filename=None):
        if self.cfg.progress_feature.split('-')[-1] == 'KL':
            if self.cfg.multi_progression:
                KL_df = self.multi_KL_progression()
            else:
                KL_df = self.KL_progression()
            JSW_df = self.parsing_JSW(mode='landscape')
            JSW_df.replace({'SIDE': {'L': 0, 'R': 1}}, inplace=True)
        elif self.cfg.progress_feature.split('-')[-1] == 'JSW':
            if self.cfg.multi_progression:
                JSW_df = self.multi_JSW_progression()
            else:
                JSW_df = self.JSW_progression()
            KL_df = self.parsing_KL_grade(mode='landscape')
            KL_df.replace({'SIDE': {'L': 0, 'R': 1}}, inplace=True)
        # JSW_df.rename(columns={'Progress':'Progress_JSW'}, inplace=True)
        # KL_cols = [col for col in KL_df.columns if 'KL' in col]
        # KL_cols.extend(['ID', 'SIDE', 'Progress'])
        # shorten_KL_df = KL_df[KL_cols]
        # shorten_KL_df.rename(columns={'Progress': 'Progress_KL'}, inplace=True)
        merge_KL_JSW_df = pd.merge(JSW_df, KL_df, on=['ID', 'SIDE'], how='inner')
        merge_KL_JSW_df = merge_KL_JSW_df.drop_duplicates(subset=['ID', 'SIDE'])
        if data_filename is not None:
            merge_KL_JSW_df.to_csv(os.path.join(self.save_path, f'{data_filename}.csv'), index=False)
            merge_KL_JSW_df.to_pickle(os.path.join(self.save_path, f'{data_filename}.pkl'), protocol=4)

        return merge_KL_JSW_df

def create_image_path_dataframe(cfg, mode='normal'):
    save_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    # if not os.path.exists(os.path.join(save_path, f'Image_path_{cfg.data_level}.csv')):
    img_path = glob.glob(os.path.join(cfg.root_path, f'{cfg.images_path}/*'))
    data_img = []
    for img_ROI in tqdm(img_path, total=len(img_path)):
        id = img_ROI.split(os.sep)[-1].split('_')[0]
        visit = img_ROI.split(os.sep)[-1].split('_')[1]
        position = img_ROI.split(os.sep)[-1].split('_')[-1].split('.')[0]
        path = img_ROI.split(cfg.root_path)[1]
        row_dict_ROI = {}
        row_dict_ROI['ID'] = int(id)
        row_dict_ROI['Visit'] = int(visit)
        row_dict_ROI['SIDE'] = position
        row_dict_ROI['Path'] = path

        data_img.append(row_dict_ROI)
    df_img = pd.DataFrame(data=data_img)
    df_img = df_img.replace({'SIDE': {'L': 0, 'R': 1}})  # convert SIDE to number
    df = df_img
    if mode == 'landscape':
        list_ID_unique = df_img.ID.unique()
        data = []
        for i in tqdm(list_ID_unique):
            row_id_data = df_img[df_img['ID'] == i].reset_index(drop=True)
            list_SIDE_unique = row_id_data.SIDE.unique()
            if len(list_SIDE_unique) > 1:
                for s in range(2):
                    row_dict = {}
                    row_side_data = row_id_data[row_id_data['SIDE'] == s].reset_index(drop=True)
                    list_visit_unique = sorted(row_side_data.Visit.unique())
                    for j in list_visit_unique:
                        if cfg.data_level == 'patient':
                            row_dict['ID'] = i
                            row_dict[f'IMG_{s}@{j}'] = row_side_data[row_side_data['Visit'] == j].iloc[0]['Path']
                        elif cfg.data_level == 'knee':
                            row_dict['ID'] = i
                            row_dict[f'SIDE'] = s
                            row_dict[f'IMG@{j}'] = row_side_data[row_side_data['Visit'] == j].iloc[0]['Path']
                    data.append(row_dict)
            else:
                row_dict = {}
                s = row_id_data['SIDE'][0]
                list_visit_unique = sorted(row_id_data.Visit.unique())
                for j in list_visit_unique:
                    if cfg.data_level == 'patient':
                        row_dict['ID'] = i
                        row_dict[f'IMG_{s}@{j}'] = row_id_data[row_id_data['Visit'] == j].iloc[0]['Path']
                    elif cfg.data_level == 'knee':
                        row_dict['ID'] = i
                        row_dict[f'SIDE'] = s
                        row_dict[f'IMG@{j}'] = row_id_data[row_id_data['Visit'] == j].iloc[0]['Path']

                data.append(row_dict)
        df_img_landscape = pd.DataFrame(data)
        df_img_landscape.to_csv(os.path.join(save_path, f'Image_path_{cfg.data_level}.csv'), index=False)
        df = df_img_landscape

    return df


def create_image_path_and_progression_dataframe(cfg, progression_filename):
    data_progression_path = os.path.join(os.path.join(cfg.root_path,cfg.dataframe_path),f'{progression_filename}.csv')
    if os.path.exists(data_progression_path):
        df_progression = pd.read_csv(data_progression_path)
    else:
        raise ValueError(f'Please create progression data first..')
    #Select KL collums and melt them to visit col
    # encode_visit = [0, 12, 24, 36, 48, 72, 96]
    # fixed_cols = ['ID','SIDE','Site','Progress']
    # KL_cols = [x for x in df_progression.columns if 'KL' in x]
    # selected_cols = fixed_cols + KL_cols
    # df_progression = df_progression[selected_cols]
    # df_KL_prog = pd.DataFrame()
    # for i in range(7):
    #     cols = fixed_cols + [f'KL@{encode_visit[i]}']
    #     df_visit = df_progression[cols].rename(columns={f'KL@{encode_visit[i]}': 'KL'})
    #     df_visit['Visit'] = int(encode_visit[i])
    #     df_KL_prog = pd.concat([df_KL_prog,df_visit],axis=0)

    #create dataframe from images folder
    img_path = glob.glob(os.path.join(cfg.root_path, f'{cfg.images_path}/*'))
    data_img = []
    for img_ROI in tqdm(img_path, total=len(img_path)):
        id = img_ROI.split(os.sep)[-1].split('_')[0]
        visit = img_ROI.split(os.sep)[-1].split('_')[1]
        position = img_ROI.split(os.sep)[-1].split('_')[-1].split('.')[0]
        path = img_ROI.split(cfg.root_path)[1]
        row_dict_ROI = {}
        row_dict_ROI['ID'] = int(id)
        row_dict_ROI['Visit'] = int(visit)
        row_dict_ROI['SIDE'] = position
        row_dict_ROI['Path'] = path

        data_img.append(row_dict_ROI)
    df_img = pd.DataFrame(data=data_img)
    df_img = df_img.replace({'SIDE': {'L': 0, 'R': 1}})  #convert SIDE to number
    #merge df images path to df KL progression
    list_ID_unique = df_img.ID.unique()
    data = []
    for i in list_ID_unique:
        row_dict = {}
        row_id_data = df_img[df_img['ID'] == i].reset_index(drop=True)

        row_dict['ID'] = i
        list_SIDE_unique = row_id_data.SIDE.unique()
        if len(list_SIDE_unique) > 1:
            for s in range(2):
                row_side_data = row_id_data[row_id_data['SIDE'] == s].reset_index(drop=True)
                list_visit_unique = sorted(row_side_data.Visit.unique())
                for j in list_visit_unique:
                    row_dict[f'IMG_{s}@{j}'] = row_side_data[row_side_data['Visit'] == j].iloc[0]['Path']
        else:
            s = row_id_data['SIDE'][0]
            list_visit_unique = sorted(row_id_data.Visit.unique())
            for j in list_visit_unique:
                row_dict[f'IMG_{s}@{j}'] = row_id_data[row_id_data['Visit'] == j].iloc[0]['Path']

        data.append(row_dict)
    df_img_landscape = pd.DataFrame(data)
    print(df_img_landscape)

    df_final = pd.merge(df_progression, df_img_landscape, on=['ID'], how='inner')
    # df_final.dropna(subset=['KL'], inplace=True)
    save_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    df_final.to_csv(os.path.join(save_path, f'Image_path_progression.csv'), index=False)
    print(df_final)

def create_df_KL_probs_from_semixup(cfg, df_progression):
    dataframe_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    df_KL_probs = pd.read_pickle(os.path.join(dataframe_path, f'KL_probs.pkl'))
    df_KL_probs = df_KL_probs.replace({'SIDE': {'L': 0, 'R': 1}})  # convert SIDE to number
    list_unique_id = df_KL_probs.ID.unique()
    data = []
    for i in list_unique_id:
        df_id = df_KL_probs[df_KL_probs['ID'] == i]
        if len(df_id.SIDE.unique()) > 1:
            for s in df_id.SIDE.unique():
                df_id_side = df_id[df_id['SIDE'] == s]
                row_dict = {}
                row_dict['ID'] = i
                row_dict['SIDE'] = s
                for j in range(len(df_id_side)):
                    row_data = df_id_side.iloc[j]
                    row_dict[f'KL_Probs@{row_data["Visit"]}'] = row_data['Prob']
                data.append(row_dict)
        else:
            s = df_id.SIDE.unique()[0]
            df_id_side = df_id[df_id['SIDE'] == s]
            row_dict = {}
            row_dict['ID'] = i
            row_dict['SIDE'] = s
            for j in range(len(df_id_side)):
                row_data = df_id_side.iloc[j]
                row_dict[f'KL_Probs@{row_data["Visit"]}'] = row_data['Prob']
            data.append(row_dict)
    df_probs_landscape = pd.DataFrame(data)
    df_final = pd.merge(df_progression, df_probs_landscape, on=['ID','SIDE'], how='inner')
    df_final.to_pickle(os.path.join(dataframe_path,f'SF12_ClinicalVars_Progression_Probs.pkl'), protocol=4)

    return df_final

def create_dataframe_from_KL_prediction_fromHoang(cfg):
    dataframe_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    df_KL_preds = pd.read_csv(os.path.join(dataframe_path, f'KL_predictions.csv'))
    data = []
    for i in tqdm(range(len(df_KL_preds)),total=len(df_KL_preds), desc=f"Reframe KL prediction"):
        row_dict = {}
        filename = df_KL_preds.iloc[i]['Filename'].split('.')[0]
        row_dict['ID'] = int(filename.split('_')[0])
        row_dict['SIDE'] = filename.split('_')[2]
        row_dict['Visit'] = int(filename.split('_')[1])
        row_dict['KL'] = np.array([df_KL_preds.iloc[i][f'KL_{kl}_pred'] for kl in range(5)])
        data.append(row_dict)
    reframe_df_KL = pd.DataFrame(data)
    reframe_df_KL = reframe_df_KL.replace({'SIDE': {'L': 0, 'R': 1}})
    list_unique_id = reframe_df_KL.ID.unique()
    data = []
    for i in tqdm(list_unique_id, total=len(list_unique_id), desc=f"Create dataframe KL preds based on knee/patient level"):
        df_id = reframe_df_KL[reframe_df_KL['ID'] == i]
        if len(df_id.SIDE.unique()) > 1:
            if cfg.data_level == 'knee':
                for s in df_id.SIDE.unique():
                    row_dict = {}
                    df_id_side = df_id[df_id['SIDE'] == s]
                    row_dict['ID'] = i
                    for j in range(len(df_id_side)):
                        row_data = df_id_side.iloc[j]
                        if row_data["Visit"]/12 <= cfg.time_interval:
                            row_dict['SIDE'] = s
                            row_dict[f'KL@{row_data["Visit"]}'] = row_data['KL']
                        else:
                            pass
                    data.append(row_dict)
            elif cfg.data_level == 'patient':
                row_dict = {}
                row_dict['ID'] = i
                for s in df_id.SIDE.unique():
                    df_id_side = df_id[df_id['SIDE'] == s]
                    for j in range(len(df_id_side)):
                        row_data = df_id_side.iloc[j]
                        if row_data["Visit"]/12 <= cfg.time_interval:
                                row_dict[f'KL_{s}@{row_data["Visit"]}'] = row_data['KL']
                        else:
                            pass
                data.append(row_dict)
        else:
            s = df_id.SIDE.unique()[0]
            df_id_side = df_id[df_id['SIDE'] == s]
            row_dict = {}
            row_dict['ID'] = i
            for j in range(len(df_id_side)):
                row_data = df_id_side.iloc[j]
                if row_data["Visit"] / 12 <= cfg.time_interval:
                    if cfg.data_level == 'knee':
                        row_dict['SIDE'] = s
                        row_dict[f'KL@{row_data["Visit"]}'] = row_data['KL']
                    elif cfg.data_level == 'patient':
                        row_dict[f'KL_{s}@{row_data["Visit"]}'] = row_data['KL']
                else:
                    pass
            data.append(row_dict)
    landscape_df_KL = pd.DataFrame(data)
    landscape_df_KL.dropna(inplace=True)
    landscape_df_KL.reset_index(drop=True, inplace=True)
    landscape_df_KL.to_pickle(os.path.join(dataframe_path, f'KL_predictions_landscape_{cfg.data_level}.pkl'), protocol=4)
    return landscape_df_KL
def create_dataframe_from_KL_prediction(cfg):
    dataframe_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    df_KL_preds = pd.read_pickle(os.path.join(dataframe_path, f'KL_predictions.pkl'))
    if cfg.progressor_type == 'prob':
        col = 'Probs'
    elif cfg.progressor_type == 'logit':
        col = 'Logits'
    elif cfg.progressor_type == 'feature':
        col = 'Features'
    else:
        raise ValueError(f'Not supported')
    data = []
    for i in tqdm(range(len(df_KL_preds)),total=len(df_KL_preds), desc=f"Reframe KL prediction"):
        row_dict = {}
        filename = df_KL_preds.iloc[i]['Filename'].split('.')[0]
        row_dict['ID'] = int(filename.split('_')[0])
        row_dict['SIDE'] = filename.split('_')[2]
        row_dict['Visit'] = int(filename.split('_')[1])
        row_dict['KL'] = df_KL_preds.iloc[i][f'{col}']
        data.append(row_dict)
    reframe_df_KL = pd.DataFrame(data)
    reframe_df_KL = reframe_df_KL.replace({'SIDE': {'L': 0, 'R': 1}})
    list_unique_id = reframe_df_KL.ID.unique()
    data = []
    for i in tqdm(list_unique_id, total=len(list_unique_id), desc=f"Create dataframe KL preds based on knee/patient level"):
        df_id = reframe_df_KL[reframe_df_KL['ID'] == i]
        if len(df_id.SIDE.unique()) > 1:
            if cfg.data_level == 'knee':
                for s in df_id.SIDE.unique():
                    row_dict = {}
                    df_id_side = df_id[df_id['SIDE'] == s]
                    row_dict['ID'] = i
                    for j in range(len(df_id_side)):
                        row_data = df_id_side.iloc[j]
                        if row_data["Visit"]/12 <= cfg.time_interval:
                            row_dict['SIDE'] = s
                            row_dict[f'KL@{row_data["Visit"]}'] = row_data['KL']
                        else:
                            pass
                    data.append(row_dict)
            elif cfg.data_level == 'patient':
                row_dict = {}
                row_dict['ID'] = i
                for s in df_id.SIDE.unique():
                    df_id_side = df_id[df_id['SIDE'] == s]
                    for j in range(len(df_id_side)):
                        row_data = df_id_side.iloc[j]
                        if row_data["Visit"]/12 <= cfg.time_interval:
                                row_dict[f'KL_{s}@{row_data["Visit"]}'] = row_data['KL']
                        else:
                            pass
                data.append(row_dict)
        else:
            s = df_id.SIDE.unique()[0]
            df_id_side = df_id[df_id['SIDE'] == s]
            row_dict = {}
            row_dict['ID'] = i
            for j in range(len(df_id_side)):
                row_data = df_id_side.iloc[j]
                if row_data["Visit"] / 12 <= cfg.time_interval:
                    if cfg.data_level == 'knee':
                        row_dict['SIDE'] = s
                        row_dict[f'KL@{row_data["Visit"]}'] = row_data['KL']
                    elif cfg.data_level == 'patient':
                        row_dict[f'KL_{s}@{row_data["Visit"]}'] = row_data['KL']
                else:
                    pass
            data.append(row_dict)
    landscape_df_KL = pd.DataFrame(data)
    landscape_df_KL.dropna(inplace=True)
    landscape_df_KL.reset_index(drop=True, inplace=True)
    landscape_df_KL.to_pickle(os.path.join(dataframe_path, f'KL_predictions_landscape_{cfg.data_level}_{cfg.progressor_type}.pkl'), protocol=4)
    return landscape_df_KL
def merge_df_KL_preds_to_processed_data(cfg, df):
    dataframe_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    if not os.path.exists(os.path.join(dataframe_path, f'KL_predictions_landscape_{cfg.data_level}_{cfg.progressor_type}.pkl')):
        df_KL = create_dataframe_from_KL_prediction(cfg)
    else:
        df_KL = pd.read_pickle(os.path.join(dataframe_path, f'KL_predictions_landscape_{cfg.data_level}_{cfg.progressor_type}.pkl'))
    if not os.path.exists(
            os.path.join(dataframe_path, f'KL_predictions_ID.pkl')):
        df_KL[['ID']].to_pickle(os.path.join(dataframe_path, f'KL_predictions_ID.pkl'), protocol=4)
    # KL_cols = [col for col in df.columns if 'KL' in col]
    # df = df.drop(KL_cols, axis=1)
    if cfg.data_level == 'patient':
        final_df = pd.merge(df, df_KL, on=['ID'], how='inner')
    elif cfg.data_level == 'knee':
        list_ID_only_one_knee = [r for r in df_KL.ID.unique() if
                                 len(df_KL[df_KL['ID'] == r]) == 1]
        df_KL = df_KL[~df_KL['ID'].isin(list_ID_only_one_knee)].reset_index(drop=True)
        final_df = pd.merge(df, df_KL, on=['ID','SIDE'], how='inner')
    # final_df.to_pickle(os.path.join(dataframe_path, f'Post_Processing_with_KL_preds.pkl'), protocol=4)
    # final_df.to_csv(os.path.join(dataframe_path, f'Post_Processing_with_KL_preds.csv'))

    return final_df




@hydra.main(config_path=os.pardir, config_name="config.yaml")
def main(cfg):
    # root_path = '/users/khanhnguyen/Data/'
    # OAI_metadata_folder = 'OAICompleteData_ASCII/'
    # OAI_extracted_data_folder = 'OAIExtractedData/'
    # Clinical_file = 'Clinical_data_follow_ups.csv'
    # KL_var_file = 'ID_SIDE_KL.csv'
    #
    # metadata_path = f'{root_path}{OAI_metadata_folder}'
    # save_path = f'{root_path}{OAI_extracted_data_folder}'
    # OAI_data = ParseMetaDataOAI(cfg)
    # df = OAI_data.parsing_KL_grade()
    cfg.dataframe_path = f'Dataframe/temp/saved_predictions_by_BA'
    cfg.progressor_type = 'other'
    cfg.data_level = 'knee'
    df = create_dataframe_from_KL_prediction(cfg)
    print(type(df['KL@0'].iloc[0]))



    # create_image_path_and_progression_dataframe(cfg, 'Post_Processing' )
    # dataframe_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    # df_post_processing = pd.read_pickle(os.path.join(dataframe_path, f'Post_processing.pkl'))
    # create_dataframe_from_KL_prediction(cfg, df_post_processing)

if __name__ == "__main__":
    main()