def build_single_img_based_multi_progression_meta(oai_src_dir, only_baseline=False, use_sas=False):
    # visits =     ['00', '12', '24', '36', '72', '96']
    # exam_codes = ['00', '01', '03', '05', '08', '10']

    visits = ['00', '12', '24', '36', '48', '72', '96']
    exam_codes = ['00', '01', '03', '05', '06', '08', '10']

    # visits =     ['48', '72', '96']
    # exam_codes = ['06', '08', '10']

    non_progressor_code = len(visits)

    grading_types = ['KL', 'OSTL', 'OSTM', 'OSFL', 'OSFM', 'JSL', 'JSM']
    max_grades = {'KL': 4, 'OSTL': 3, 'OSTM': 3, 'OSFL': 3, 'OSFM': 3, 'JSL': 3, 'JSM': 3}
    include_TKR = True
    grad_files = []
    if not os.path.isfile("oai_master_df.csv") or not os.path.isfile("oai_grad_files.pkl"):

        for i, visit in enumerate(visits):
            print(f'==> Reading OAI {visit} visit')

            if use_sas:
                meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                                      'Semi-Quant Scoring_SAS',
                                                      f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
            #     data_clinical = read_sas7bdata_pd(os.path.join(oai_src_dir, f'allclinical{exam_codes[i]}.sas7bdat'))
            else:
                meta = pd.read_csv(os.path.join(oai_src_dir,
                                                'Semi-Quant Scoring_ASCII',
                                                f'kxr_sq_bu{exam_codes[i]}.txt'), sep='|')
            #     # data_clinical = pd.read_csv(os.path.join(oai_src_dir, 'AllClinical_ASCII', f'AllClinical{exam_codes[i]}.txt'), sep='|')
            data_clinical = build_clinical(oai_src_dir, visit_id=exam_codes[i], use_sas=True)

            meta['ID'] = meta['ID'].astype(str)
            meta.replace({'SIDE': {1: sides[1], 2: sides[2]}}, inplace=True)
            data_clinical['ID'] = data_clinical['ID'].astype(str)
            data_clinical.rename(columns={'Side': 'SIDE'}, inplace=True)

            meta = meta.merge(data_clinical, on=['ID', 'SIDE'], how='left')
            # Dropping the data from multiple projects
            meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
            meta.fillna(-1, inplace=True)
            for c in meta.columns:
                meta[c.upper()] = meta[c]
            # Removing the TKR and KL4 at the baseline
            if i == 0:
                meta = meta[meta[f'V{exam_codes[i]}XRKL'] != -1]
                # meta = meta[meta[f'V{exam_codes[i]}XRKL'] < 4]
            # meta = meta[meta[f'V{exam_codes[i]}XRKL'] <= max_grades['KL']]

            meta['KL'] = meta[f'V{exam_codes[i]}XRKL']
            meta['OSTL'] = meta[f'V{exam_codes[i]}XROSTL']
            meta['OSTM'] = meta[f'V{exam_codes[i]}XROSTM']

            meta['OSFL'] = meta[f'V{exam_codes[i]}XROSFL']
            meta['OSFM'] = meta[f'V{exam_codes[i]}XROSFM']

            meta['JSL'] = meta[f'V{exam_codes[i]}XRJSL']
            meta['JSM'] = meta[f'V{exam_codes[i]}XRJSM']

            # Merge levels 0 into 1 in every grading
            # Let TKR be the highest level
            for grading in grading_types:
                meta[grading] = meta[grading].round(decimals=0)
                if grading == "KL":
                    # Remove invalid records that are out of upper bound
                    # meta = meta[meta[grading] <= max_grades[grading]]
                    meta.loc[meta[grading] > max_grades[grading], grading] = None

                    # Set TKR the highest level
                    if include_TKR:
                        meta.loc[meta[grading] == -1, grading] = max_grades[grading] + 1
                        max_kl = max_grades[grading] + 1
                    else:
                        max_kl = max_grades[grading]

                    # Remove invalid records that are out of lower bound
                    # meta = meta[meta[grading] >= 0]
                    meta.loc[meta[grading] < 0, grading] = None

                    for v in range(max_kl):
                        meta = meta.replace({grading: {v + 1: v}})
                else:
                    meta.loc[meta[grading] < 0, grading] = None

            if i == 0:
                _pain_key = f'P01'
            else:
                _pain_key = f'V{exam_codes[i]}'

            # meta['KPL30CV'] = meta[f'{_pain_key}KPL30CV']
            # meta['KPR30CV'] = meta[f'{_pain_key}KPR30CV']

            # Add visit columns
            meta['visit'] = int(visit)
            meta['visit_id'] = int(exam_codes[i])

            # meta['LKDEFCV'] = meta[f'V{exam_codes[i]}lkdefcv']
            # meta['RKDEFCV'] = meta[f'V{exam_codes[i]}rkdefcv']

            # meta['LKABPN'] = meta[f'V{exam_codes[i]}LKABPN']
            # meta['LKLTTPN'] = meta[f'V{exam_codes[i]}LKLTTPN']
            # meta['LKMTTPN'] = meta[f'V{exam_codes[i]}LKMTTPN']
            # meta['LKRFXPN'] = meta[f'V{exam_codes[i]}LKRFXPN']
            #
            # meta['RKABPN'] = meta[f'V{exam_codes[i]}RKABPN']
            # meta['RKLTTPN'] = meta[f'V{exam_codes[i]}RKLTTPN']
            # meta['RKMTTPN'] = meta[f'V{exam_codes[i]}RKMTTPN']
            # meta['RKRFXPN'] = meta[f'V{exam_codes[i]}RKRFXPN']

            grad_files.append(
                meta[['ID', 'SIDE', 'KL', 'visit', 'visit_id', 'OSTL', 'OSTM', 'OSFL', 'OSFM', 'JSL', 'JSM',
                      'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'V00SITE'
                      # , 'KPL30CV', 'KPR30CV'
                      # , 'HEIGHT', 'WEIGHT', 'KPNL12', 'KPNR12', 'LKDEFCV', 'RKDEFCV'
                      # , 'LKABPN', 'LKLTTPN', 'LKMTTPN', 'LKRFXPN',
                      #   'RKABPN', 'RKLTTPN', 'RKMTTPN', 'RKRFXPN']])
                      ]])

        id_set_last_fu = set(grad_files[-1].ID.values.astype(int).tolist())  # Subjects present at all FU

        master_df = pd.concat(grad_files)

        with open("oai_grad_files.pkl", "wb") as f:
            pickle.dump(grad_files, f, protocol=4)
        master_df.to_csv("oai_master_df.csv", index=None)
    else:
        print(f'Loading oai_grad_files.pkl')
        with open("oai_grad_files.pkl", "rb") as f:
            grad_files = pickle.load(f)
        master_df = pd.read_csv("oai_master_df.csv")

    master_df['ID'] = master_df['ID'].astype(str)
    # Get baseline and follow-ups
    # for follow_up_id in range(0, len(KL_files)):
    #     KL_files[follow_up_id] = KL_files[follow_up_id].set_index(['ID', 'SIDE'])

    # master_df = master_df.set_index(['ID', 'SIDE'])

    # looking for progressors
    identified_prog = set()

    fus_df = []

    n_bs_knees = grad_files[0].shape[0]

    for bs_knee_id, knee in tqdm(grad_files[0].iterrows(), total=n_bs_knees, desc='Processing OAI:'):
        # if len(short_tri_fus_df) > 5:
        #     break
        # if str(knee.ID) == '9015363':
        #     print('abc')
        # else:
        #     continue

        if int(knee.ID) in identified_prog:
            if identified_prog[int(knee.ID)] == sides[int(knee.SIDE)]:
                continue
        # If not healthy
        # if not check_healthy_knee(knee):
        #     identified_prog.update({(int(knee.ID), sides[int(knee.SIDE)]), })
        #     continue

        # master_df.index.isin([knee.ID, knee.SIDE])
        participant = master_df[(master_df['ID'] == knee.ID) & (master_df['SIDE'] == knee.SIDE)]
        participant.sort_values(by=['visit_id'])

        assert len(participant) > 0

        n_follow_ups = len(participant.index)

        for fu1_id in range(n_follow_ups - 1):
            valid_gradings1 = get_valid_gradings_1st_fu(participant.iloc[fu1_id], grading_types, max_grades)

            fu1 = participant.iloc[fu1_id]  # .add_suffix('1')

            # if fu1['ID'] == '9009927':
            #     print('abc')

            # Condition if forced to use baseline only
            if only_baseline:
                fu1_cond = fu1['visit'] == 0
            else:
                fu1_cond = True

            if len(valid_gradings1) == 0:
                print(f'No valid gradings in FU1.')
                continue

            if not fu1_cond:
                print(f'Skip as the initial data is not a baseline.')
                continue

            # fu1['SIDE'] = sides[int(fu1['SIDE'])]
            start_visit = fu1['visit']

            fus = fu1

            valids = []
            for grade_id, grading in enumerate(grading_types):
                prev_grade = fu1[f'{grading}']

                reach_max = False
                valid_by_grading = True
                fus_by_grading = None
                for fu2_id in range(fu1_id + 1, n_follow_ups):
                    fu2 = participant.iloc[fu2_id].add_suffix('2')
                    dt12_y = int((fu2['visit2'] - start_visit) // 12)

                    if not check_valid_grading(fu2[f'{grading}2']):
                        continue

                    # Time points after a time point with max grade must be graded the max value
                    if reach_max or prev_grade == max_grades[grading]:
                        reach_max = True
                        fu2[f'{grading}2'] = max_grades[grading]

                    valid_bl = check_progression_validity(fu1[f'{grading}'], fu2[f'{grading}2'])
                    valid_prev = check_progression_validity(prev_grade, fu2[f'{grading}2'])
                    if not valid_bl or not valid_prev:
                        print(
                            f'[Invalid] {grading}, BL: {fu1[grading]}, prev: {prev_grade}, cur: {fu2[f"{grading}2"]}.')
                        valid_by_grading = False
                        break
                    if prev_grade == 1.0 and fu2[f"{grading}2"] == 0.0:
                        print(f'{grading}, BL: {fu1[grading]}, prev: {prev_grade}, cur: {fu2[f"{grading}2"]}.')

                    new_cols = {}
                    new_cols[f'DT_{dt12_y}y'] = (fu2['visit2'] - start_visit) / 12.0
                    new_cols[f'{grading}_{dt12_y}y'] = fu2[f'{grading}2']
                    new_cols = pd.Series(new_cols)

                    if fus_by_grading is None:
                        fus_by_grading = new_cols
                    else:
                        fus_by_grading = pd.concat([fus_by_grading, new_cols])

                    if check_valid_grading(fu2[f'{grading}2']):
                        prev_grade = fu2[f'{grading}2']

                if not valid_by_grading:
                    fus_by_grading = None

                # End fu2 loop
                if fus_by_grading is not None:
                    fus = pd.concat([fus, fus_by_grading])
                valids.append(valid_by_grading)

            # End grading loop
            if any(valids):
                fus_df.append(fus.to_dict())

    print(f'\nConverting to dataframes...')
    fus_df = pd.DataFrame(fus_df)

    # Remove redundant columns
    # Rename ID and Side
    fus_df = fus_df.rename(columns={'SIDE': 'Side'})
    if 'IDs' in fus_df:
        fus_df = fus_df.drop(columns=['ID2'])
    if 'SIDE2' in fus_df:
        fus_df = fus_df.drop(columns=['SIDE2'])

    fus_df = fus_df.astype({'ID': str})

    return fus_df