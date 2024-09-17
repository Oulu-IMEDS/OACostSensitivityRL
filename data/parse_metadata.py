import hydra
import pandas as pd
import glob
import os
from tqdm import tqdm
from sas7bdat import SAS7BDAT
import numpy as np
import matplotlib.pyplot as plt

class ParseMetaDataOAI():
    def __init__(self, cfg):
        self.path = os.path.join(cfg.root_path, cfg.metadata_path)
        self.follow_ups = ['00', '01', '03', '05', '06', '08', '10']
        self.encode_visit = {'00': 0, '01': 12, '03': 24, '05': 36, '06': 48, '08': 72, '10': 96}
        self.follow_ups_len = len(self.follow_ups)

    def parsing_JSW(self, mode='normal'):
        df_JSW_final = pd.DataFrame(columns=['ID', 'SIDE'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            visit = self.encode_visit[id]
            file_data = pd.read_table(os.path.join(self.path, f'kxr_qjsw_duryea{id}.txt'), delimiter='|')

            #Medial Joint space width
            MCMJSW = f'V{id}MCMJSW' #min
            JSW150 = f'V{id}JSW150'
            JSW175 = f'V{id}JSW175'
            JSW200 = f'V{id}JSW200'
            JSW225 = f'V{id}JSW225'
            JSW250 = f'V{id}JSW250'
            JSW275 = f'V{id}JSW275'
            JSW300 = f'V{id}JSW300'

            #Lateral joint space width
            LJSW700 = f'V{id}LJSW700'
            LJSW725 = f'V{id}LJSW725'
            LJSW750 = f'V{id}LJSW750'
            LJSW775 = f'V{id}LJSW775'
            LJSW800 = f'V{id}LJSW800'
            LJSW825 = f'V{id}LJSW825'
            LJSW850 = f'V{id}LJSW850'
            LJSW875 = f'V{id}LJSW875'
            LJSW900 = f'V{id}LJSW900'

            selected_headers = ['ID',MCMJSW, JSW150, JSW175, JSW200, JSW225,
                                JSW250, JSW275, JSW300, LJSW700, LJSW725,
                                LJSW750, LJSW775, LJSW800, LJSW825, LJSW850,
                                LJSW875, LJSW900]
            based_col = [f'minJSW', f'MJSW150', f'MJSW175', f'MJSW200', f'MJSW225'
                , f'MJSW250', f'MJSW275', f'MJSW300', f'LJSW700', f'LJSW725'
                , f'LJSW750', f'LJSW775', f'LJSW800', f'LJSW825', f'LJSW850'
                , f'LJSW875', f'LJSW900']

            if f'side' in file_data.columns:
                selected_headers.insert(1, 'side')
                df_JSW = file_data[selected_headers]
                df_JSW = df_JSW.rename(columns={'side': 'SIDE'})
            elif f'Side' in file_data.columns:
                selected_headers.insert(1, 'Side')
                df_JSW = file_data[selected_headers]
                df_JSW = df_JSW.rename(columns={'Side': 'SIDE'})
            else:
                selected_headers.insert(1, 'SIDE')
                df_JSW = file_data[selected_headers]


            if mode == 'landscape':
                names_col = [s + f'@{visit}' for s in based_col]
                start_count = len(selected_headers) - len(names_col)
                for j in range(start_count,len(selected_headers)):
                    df_JSW.rename(columns={f'{selected_headers[j]}': f'{names_col[j-start_count]}'}, inplace=True)

                df_JSW.ID = df_JSW.ID.values.astype(int)
                df_JSW_final = pd.merge(df_JSW_final, df_JSW, on=['ID', 'SIDE'], how='outer')
            elif mode == 'normal':
                df_JSW['Visit'] = visit
                names_col = based_col
                start_count = len(selected_headers) - len(names_col)
                for j in range(start_count, len(selected_headers)):
                    df_JSW.rename(columns={f'{selected_headers[j]}': f'{names_col[j - start_count]}'}, inplace=True)

                df_JSW.ID = df_JSW.ID.values.astype(int)
                df_JSW.Visit = df_JSW.Visit.values.astype(int)
                df_JSW_final = pd.concat([df_JSW_final, df_JSW])
            else:
                raise ValueError(f'only support mode normal/landscape')
            # df_JSW = pd.concat([df_JSW,kXR])
        df_JSW_final['SIDE'] = df_JSW_final['SIDE'].map({1: 'R', 2: 'L'})
        return df_JSW_final

    def parsing_WOMAC_score(self, mode='normal'):
        df_WOMAC_final = pd.DataFrame(columns=['ID', 'SIDE'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            file_data = pd.read_table(os.path.join(self.path, f'AllClinical{id}.txt'), delimiter='|')

            # Visit, Site
            df_WOMAC = file_data[['ID']]
            df_WOMAC.ID = df_WOMAC.ID.values.astype(int)

            # Making side-wise metadata
            df_WOMAC_left = df_WOMAC.copy()
            df_WOMAC_right = df_WOMAC.copy()
            df_WOMAC_left['SIDE'] = 'L'
            df_WOMAC_right['SIDE'] = 'R'

            # Disability score V{01}WOMADL(L/R)
            # Pain score V{01}WOMKP(L/R)
            # Stiffness score V{01}WOMSTF(L/R)
            # Total score V{01}WOMTS(L/R)
            headers = [f'V{id}WOMADL', f'V{id}WOMKP', f'V{id}WOMSTF', f'V{id}WOMTS']
            if mode == 'normal':
                df_WOMAC['Visit'] = self.encode_visit[id]
                name_cols = ['WOMACFunc','WOMACPain','WOMACStiff','WOMAC']
                for index, header in enumerate(headers):
                    df_WOMAC_left[f'{name_cols[index]}'] = file_data[f'{header}L']
                    df_WOMAC_right[f'{name_cols[index]}'] = file_data[f'{header}R']

                df_WOMAC = pd.concat((df_WOMAC_left, df_WOMAC_right))
                df_WOMAC.dropna(inplace=True)

                df_WOMAC_final = pd.concat([df_WOMAC_final, df_WOMAC])
            elif mode == 'landscape':
                visit = self.encode_visit[id]
                name_cols = [f'WOMACFunc@{visit}',f'WOMACPain@{visit}',f'WOMACStiff@{visit}',f'WOMAC@{visit}']
                for index, header in enumerate(headers):
                    df_WOMAC_left[f'{name_cols[index]}'] = file_data[f'{header}L']
                    df_WOMAC_right[f'{name_cols[index]}'] = file_data[f'{header}R']

                df_WOMAC = pd.concat((df_WOMAC_left, df_WOMAC_right))
                df_WOMAC.dropna(inplace=True)

                df_WOMAC_final = pd.merge(df_WOMAC_final, df_WOMAC, on=['ID', 'SIDE'], how='outer')

            else:
                raise ValueError(f'only support mode normal/landscape')


        return df_WOMAC_final

    def parsing_PA_score(self, mode='normal'):
        df_PA_final = pd.DataFrame(columns=['ID'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            file_data = pd.read_table(os.path.join(self.path, f'AllClinical{id}.txt'), delimiter='|')
            # Visit
            visit = self.encode_visit[id]
            df_PA = file_data[['ID']]
            df_PA.ID = df_PA.ID.values.astype(int)

            # Physical Activity for elderly score E
            if mode == 'normal':
                df_PA['Visit'] = visit
                df_PA['PAS'] = file_data[f'V{id}PASE']
                df_PA.dropna(inplace=True)

                df_PA_final = pd.concat([df_PA_final, df_PA])
            elif mode == 'landscape':
                df_PA[f'PAS@{visit}'] = file_data[f'V{id}PASE']
                df_PA.dropna(inplace=True)
                df_PA_final = pd.merge(df_PA_final, df_PA, on=['ID'], how='outer')

            else:
                raise ValueError(f'only support mode normal/landscape')

        return df_PA_final

    def parsing_KOOSQoL_score(self, mode='normal'):
        df_KOOSQoL_final = pd.DataFrame(columns=['ID'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            file_data = pd.read_table(os.path.join(self.path, f'AllClinical{id}.txt'), delimiter='|')
            # Visit
            df_KOOSQoL = file_data[['ID']]
            visit = self.encode_visit[id]
            df_KOOSQoL.ID = df_KOOSQoL.ID.values.astype(int)

            # KOOS Quality of Life
            if mode == 'normal':
                df_KOOSQoL['Visit'] = visit
                df_KOOSQoL['KOOSQoL'] = file_data[f'V{id}KOOSQOL']
                df_KOOSQoL.dropna(inplace=True)

                df_KOOSQoL_final = pd.concat([df_KOOSQoL_final, df_KOOSQoL])
            elif mode == 'landscape':
                df_KOOSQoL[f'KOOSQoL@{visit}'] = file_data[f'V{id}KOOSQOL']
                df_KOOSQoL.dropna(inplace=True)

                df_KOOSQoL_final = pd.merge(df_KOOSQoL_final, df_KOOSQoL, on=['ID'], how='outer')
            else:
                raise ValueError(f'only support mode normal/landscape')

        return df_KOOSQoL_final

    def parsing_patient_info(self, mode='normal'):
        df_patient_final = pd.DataFrame(columns=['ID'])
        enrollees_data = self.read_sas7bdata_pd(os.path.join(self.path, f'enrollees.sas7bdat'))
        enrollees_data = enrollees_data[['ID', 'V00SITE','P02SEX']]
        enrollees_data.ID = enrollees_data.ID.values.astype(int)
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            file_data = pd.read_table(os.path.join(self.path, f'AllClinical{id}.txt'), delimiter='|')

            df_patient = file_data[['ID']]
            df_patient.ID = df_patient.ID.values.astype(int)

            if id == '00':
                AGE_col = 'V00AGE'
                BMI_col = 'P01BMI'
                HEIGHT_col = 'P01HEIGHT'
                WEIGHT_col = 'P01WEIGHT'
            else:
                AGE_col = f'V{id}AGE'
                BMI_col = f'V{id}BMI'
                HEIGHT_col = f'V{id}HEIGHT'
                WEIGHT_col = f'V{id}WEIGHT'


            # AGE, BMI, HEIGHT, WEIGHT
            headers = [f'{AGE_col}', f'{BMI_col}', f'{WEIGHT_col}']
            if mode == 'normal':
                df_patient['Visit'] = self.encode_visit[id]
                name_cols = ['AGE','BMI','WEIGHT']
                for index, header in enumerate(headers):
                    df_patient[f'{name_cols[index]}'] = file_data[f'{header}']

                if f'{HEIGHT_col}' in file_data.columns:
                    df_patient['HEIGHT'] = file_data[f'{HEIGHT_col}']
                else:
                    df_height = df_patient_final[
                        (df_patient_final['Visit'] == self.encode_visit[self.follow_ups[i - 1]])][
                        ['ID', 'HEIGHT']]
                    df_patient = pd.merge(df_patient, df_height, on=['ID'], how='left')

                df_patient.dropna(subset=['BMI'],inplace=True)
                df_patient_final = pd.concat([df_patient_final, df_patient])
                df_patient_final.Visit = df_patient_final.Visit.values.astype(int)
            elif mode == 'landscape':
                visit = self.encode_visit[self.follow_ups[i]]
                name_cols = [f'AGE@{visit}', f'BMI@{visit}', f'WEIGHT@{visit}']
                for index, header in enumerate(headers):
                    df_patient[f'{name_cols[index]}'] = file_data[f'{header}']

                if f'{HEIGHT_col}' in file_data.columns:
                    df_patient[f'HEIGHT@{visit}'] = file_data[f'{HEIGHT_col}']
                else:
                    df_height = df_patient_final[['ID', f'HEIGHT@{self.encode_visit[self.follow_ups[i-1]]}']]
                    df_patient = pd.merge(df_patient, df_height, on=['ID'], how='left')
                    df_patient.rename(columns={f'HEIGHT@{self.encode_visit[self.follow_ups[i-1]]}':f'HEIGHT@{visit}'},inplace=True)

                df_patient.dropna(subset=[f'BMI@{visit}'], inplace=True)
                df_patient_final = pd.merge(df_patient_final, df_patient, on=['ID'], how='outer')

            else:
                raise ValueError(f'only support mode normal/landscape')

        df_patient_final = self.fill_age_cols(df_patient_final, mode)
        df_patient_final = pd.merge(df_patient_final, enrollees_data, on='ID', how='left')
        df_patient_final.rename(index=str, columns={'V00SITE': 'Site', 'P02SEX': 'SEX'}, inplace=True)
        cols = df_patient_final.columns.tolist()
        cols.remove('Site')
        cols.insert(1, 'Site')
        df_patient_final = df_patient_final[cols]


        return df_patient_final

    def fill_age_cols(self, df, mode='normal'):

        if mode == 'normal':
            df_filled_age = df[df['Visit']==0]
            for i in range(1, self.follow_ups_len):
                age_baseline = df[df['Visit']==0]['AGE']
                df_visit = df[df['Visit']==self.encode_visit[self.follow_ups[i]]]
                df_visit['AGE'] = age_baseline + int(self.encode_visit[self.follow_ups[i]]/12)
                df_filled_age = pd.concat([df_filled_age, df_visit])
        elif mode == 'landscape':
            df_filled_age = df
            for i in range(1, self.follow_ups_len):
                age_baseline = df['AGE@0']
                df_filled_age[f'AGE@{self.encode_visit[self.follow_ups[i]]}'] = age_baseline + int(self.encode_visit[self.follow_ups[i]]/12)
        else:
            raise ValueError(f'only support mode normal/landscape')

        return df_filled_age




    def parsing_KL_grade(self, mode='normal'):
        df_KL_final = pd.DataFrame(columns=['ID', 'SIDE'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            visit = self.encode_visit[id]
            file_data = pd.read_table(os.path.join(self.path, f'kxr_sq_bu{id}.txt'), delimiter='|')

            if f'V{id}XRKL' in file_data.columns:
                KL_col = f'V{id}XRKL'
            elif f'v{id}XRKL' in file_data.columns:
                KL_col = f'v{id}XRKL'

            if mode=='normal':
                df_KL = file_data.filter(items=['ID', 'SIDE', KL_col]).rename(index=str, columns={KL_col: 'KL'})
                df_KL['Visit'] = self.encode_visit[id]

                df_KL_final = pd.concat([df_KL_final, df_KL])
            elif mode == 'landscape':
                df_KL = file_data.filter(items=['ID', 'SIDE', KL_col]).rename(index=str, columns={KL_col: f'KL@{visit}'})
                df_KL_final = pd.merge(df_KL_final, df_KL, on=['ID','SIDE'], how='outer')
            else:
                raise ValueError(f'only support mode normal/landscape')

        df_KL_final['SIDE'] = df_KL_final['SIDE'].map({1: 'R', 2: 'L'})
        df_KL_final.drop_duplicates(inplace=True)
        return df_KL_final

    def parsing_SF_12_score(self, mode='normal'):
        df_SF12_final = pd.DataFrame(columns=['ID'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            file_data = pd.read_table(os.path.join(self.path, f'AllClinical{id}.txt'), delimiter='|')
            # Visit
            df_SF12= file_data[['ID']]
            visit = self.encode_visit[id]
            df_SF12.ID = df_SF12.ID.values.astype(int)

            # Physical SF12 score
            if mode == 'normal':
                df_SF12['Visit'] = visit
                df_SF12['SF12'] = file_data[f'V{id}HSPSS']
                df_SF12.dropna(inplace=True)

                df_SF12_final = pd.concat([df_SF12_final, df_SF12])
            elif mode == 'landscape':
                df_SF12[f'SF12@{visit}'] = file_data[f'V{id}HSPSS']
                df_SF12.dropna(inplace=True)

                df_SF12_final = pd.merge(df_SF12_final, df_SF12, on=['ID'], how='outer')
            else:
                raise ValueError(f'only support mode normal/landscape')

        return df_SF12_final


    def parsing_other_clinical_variables(self, mode='normal'):
        df_clinical_final = pd.DataFrame(columns=['ID', 'SIDE'])
        for i in range(self.follow_ups_len):
            id = self.follow_ups[i]
            visit = self.encode_visit[id]
            file_data = self.read_sas7bdata_pd(os.path.join(self.path,f'allclinical{id}.sas7bdat'))

            # Visit, Site
            df_clinical = file_data[['ID']]
            df_clinical.ID = df_clinical.ID.values.astype(int)

            # Making side-wise metadata
            df_clinical_left = df_clinical.copy()
            df_clinical_right = df_clinical.copy()
            df_clinical_left['SIDE'] = 'L'
            df_clinical_right['SIDE'] = 'R'
            df_clinical_right['SIDE'] = 'R'

            if id == '00':
                INJ_col = 'P01INJ'
                # INJR_col = 'P01INJR'
                SURG_col = 'P01KSURG'
                # SURGR_col = 'P01KSURGR'
            else:
                INJ_col = f'V{id}INJ'
                # INJR_col = f'V{id}INJR12'
                SURG_col = f'V{id}KSRG'
                # SURGR_col = f'V{id}KSRGR12'

            headers = [f'{INJ_col}', f'{SURG_col}']
            if mode == 'normal':
                df_clinical['Visit'] = visit
                name_cols = ['INJ', 'SURG']
                for index, header in enumerate(headers):
                    if (f'{header}L' in file_data.columns) and (f'{header}R' in file_data.columns):
                        df_clinical_left[f'{name_cols[index]}'] = file_data[f'{header}L']
                        df_clinical_right[f'{name_cols[index]}'] = file_data[f'{header}R']
                    else:
                        df_clinical_left[f'{name_cols[index]}'] = file_data[f'{header}L12']
                        df_clinical_right[f'{name_cols[index]}'] = file_data[f'{header}R12']

                df_clinical = pd.concat((df_clinical_left, df_clinical_right))
                df_clinical.dropna(inplace=True)

                df_clinical_final = pd.concat([df_clinical_final, df_clinical])
            elif mode == 'landscape':
                name_cols = [f'INJ@{visit}', f'SURG@{visit}']
                for index, header in enumerate(headers):
                    if (f'{header}L' in file_data.columns) and (f'{header}R' in file_data.columns):
                        df_clinical_left[f'{name_cols[index]}'] = file_data[f'{header}L']
                        df_clinical_right[f'{name_cols[index]}'] = file_data[f'{header}R']
                    else:
                        df_clinical_left[f'{name_cols[index]}'] = file_data[f'{header}L12']
                        df_clinical_right[f'{name_cols[index]}'] = file_data[f'{header}R12']

                df_clinical = pd.concat((df_clinical_left, df_clinical_right))
                df_clinical.dropna(inplace=True)

                df_clinical_final = pd.merge(df_clinical_final, df_clinical, on=['ID', 'SIDE'], how='outer')

            else:
                raise ValueError(f'only support mode normal/landscape')


        return df_clinical_final

    def read_sas7bdata_pd(self, fname):
        data = []
        with SAS7BDAT(fname) as f:
            for row in f:
                data.append(row)

        return pd.DataFrame(data[1:], columns=data[0])


@hydra.main(config_path=os.pardir, config_name="config.yaml")
def main(cfg):
    root_path = '/Users/khanhnguyen/Data/OAI'
    OAI_metadata_folder = 'OAICompleteData_ASCII/'
    OAI_extracted_data_folder = 'OAIExtractedData/'
    Clinical_file = 'Clinical_data_follow_ups.csv'
    KL_var_file = 'ID_SIDE_KL.csv'

    path = f'{root_path}{OAI_metadata_folder}'
    OAI_data = ParseMetaDataOAI(cfg)
    df = OAI_data.parsing_SF_12_score(mode='landscape')
    print(df)
    print(len(df.ID.unique()))
    print(df[:100])

if __name__ == "__main__":
    main()

