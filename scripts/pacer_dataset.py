import pickle
import argparse
import random
import os
import os.path as osp
from tqdm import tqdm


amass_run_data = [
    '0-ACCAD_Female1Running_c3d_C25 -  side step right_poses',
    '0-ACCAD_Female1Running_c3d_C5 - walk to run_poses',
    '0-ACCAD_Female1Walking_c3d_B15 - walk turn around (same direction)_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B15 - Walk turn around_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B16 - Walk turn change_poses',
    '0-ACCAD_Male2Running_c3d_C17 - run change direction_poses',
    '0-ACCAD_Male2Running_c3d_C20 - run to pickup box_poses',
    '0-ACCAD_Male2Running_c3d_C24 - quick sidestep left_poses',
    '0-ACCAD_Male2Running_c3d_C3 - run_poses',
    '0-ACCAD_Male2Walking_c3d_B15 -  Walk turn around_poses',
    '0-ACCAD_Male2Walking_c3d_B17 -  Walk to hop to walk a_poses',
    '0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk t2_poses',
    '0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk_poses',
    '0-BioMotionLab_NTroje_rub001_0017_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub020_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub027_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub076_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub077_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub104_0027_circle_walk_poses',
    '0-Eyes_Japan_Dataset_aita_walk-04-fast-aita_poses',
    '0-Eyes_Japan_Dataset_aita_walk-21-one leg-aita_poses',
    '0-Eyes_Japan_Dataset_frederic_walk-04-fast-frederic_poses',
    '0-Eyes_Japan_Dataset_hamada_walk-06-catwalk-hamada_poses',
    '0-Eyes_Japan_Dataset_kaiwa_walk-27-thinking-kaiwa_poses',
    '0-Eyes_Japan_Dataset_shiono_walk-09-handbag-shiono_poses',
    '0-HumanEva_S2_Jog_1_poses', '0-HumanEva_S2_Jog_3_poses',
    '0-KIT_10_WalkInClockwiseCircle10_poses',
    '0-KIT_10_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_10_WalkInCounterClockwiseCircle10_poses',
    '0-KIT_12_WalkInClockwiseCircle09_poses',
    '0-KIT_12_WalkInClockwiseCircle11_poses',
    '0-KIT_12_WalkInCounterClockwiseCircle01_poses',
    '0-KIT_12_WalkingStraightForwards03_poses', '0-KIT_167_downstairs04_poses',
    '0-KIT_167_upstairs_downstairs01_poses',
    '0-KIT_167_walking_medium04_poses', '0-KIT_167_walking_run02_poses',
    '0-KIT_167_walking_run06_poses', '0-KIT_183_run04_poses',
    '0-KIT_183_upstairs10_poses', '0-KIT_183_walking_fast03_poses',
    '0-KIT_183_walking_fast05_poses', '0-KIT_183_walking_medium04_poses',
    '0-KIT_183_walking_run04_poses', '0-KIT_183_walking_run05_poses',
    '0-KIT_183_walking_run06_poses', '0-KIT_205_walking_medium04_poses',
    '0-KIT_205_walking_medium10_poses', '0-KIT_314_run02_poses',
    '0-KIT_314_run04_poses', '0-KIT_314_walking_fast06_poses',
    '0-KIT_314_walking_medium02_poses', '0-KIT_314_walking_medium07_poses',
    '0-KIT_314_walking_slow05_poses', '0-KIT_317_walking_medium09_poses',
    '0-KIT_348_walking_medium07_poses', '0-KIT_348_walking_run10_poses',
    '0-KIT_359_downstairs04_poses', '0-KIT_359_downstairs06_poses',
    '0-KIT_359_upstairs09_poses', '0-KIT_359_upstairs_downstairs03_poses',
    '0-KIT_359_walking_fast10_poses', '0-KIT_359_walking_run05_poses',
    '0-KIT_359_walking_slow02_poses', '0-KIT_359_walking_slow09_poses',
    '0-KIT_3_walk_6m_straight_line04_poses', '0-KIT_3_walking_medium07_poses',
    '0-KIT_3_walking_medium08_poses', '0-KIT_3_walking_run03_poses',
    '0-KIT_3_walking_slow08_poses', '0-KIT_424_run05_poses',
    '0-KIT_424_upstairs03_poses', '0-KIT_424_upstairs05_poses',
    '0-KIT_424_walking_fast04_poses', '0-KIT_425_walking_fast01_poses',
    '0-KIT_425_walking_fast04_poses', '0-KIT_425_walking_fast05_poses',
    '0-KIT_425_walking_medium08_poses',
    '0-KIT_4_WalkInClockwiseCircle02_poses',
    '0-KIT_4_WalkInClockwiseCircle05_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle02_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle07_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle08_poses',
    '0-KIT_513_downstairs06_poses', '0-KIT_513_upstairs07_poses',
    '0-KIT_675_walk_with_handrail_table_left003_poses',
    '0-KIT_6_WalkInClockwiseCircle04_1_poses',
    '0-KIT_6_WalkInClockwiseCircle05_1_poses',
    '0-KIT_6_WalkInCounterClockwiseCircle01_1_poses',
    '0-KIT_6_WalkInCounterClockwiseCircle10_1_poses',
    '0-KIT_7_WalkInCounterClockwiseCircle09_poses',
    '0-KIT_7_WalkingStraightForwards04_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle03_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle10_poses',
    '0-KIT_9_WalkInClockwiseCircle04_poses',
    '0-KIT_9_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_9_WalkingStraightForwards01_poses',
    '0-KIT_9_WalkingStraightForwards04_poses', '0-KIT_9_run01_poses',
    '0-KIT_9_run05_poses', '0-KIT_9_walking_medium02_poses',
    '0-KIT_9_walking_run02_poses', '0-KIT_9_walking_slow07_poses',
    '0-SFU_0005_0005_Jogging001_poses', '0-TotalCapture_s4_walking2_poses',
    '0-Transitions_mocap_mazen_c3d_crouchwalk_running_poses',
    "0-KIT_359_walking_slow10_poses",
    '0-KIT_513_downstairs07_poses', '0-KIT_9_walking_run07_poses',
    '0-KIT_183_downstairs01_poses', '0-KIT_167_downstairs05_poses', '0-TotalCapture_s3_walking1_poses',
    '0-KIT_675_walk_with_handrail_beam_right07_poses', '0-KIT_317_run02_poses', '0-KIT_348_walking_slow04_poses',
    '0-KIT_424_walking_fast03_poses', '0-KIT_11_WalkingStraightForwards03_poses', '0-KIT_3_walking_medium09_poses',
    '0-KIT_314_walking_medium06_poses', '0-SFU_0008_0008_Walking002_poses', '0-KIT_9_walking_run08_poses', '0-KIT_11_WalkingStraightForwards02_poses',
    '0-BioMotionLab_NTroje_rub021_0027_circle_walk_poses', '0-KIT_425_walking_medium09_poses', '0-KIT_348_run04_poses', '0-KIT_183_walking_fast10_poses',
    '0-KIT_424_walking_fast05_poses', '0-KIT_8_WalkInCounterClockwiseCircle01_poses', '0-KIT_317_run03_poses', '0-BioMotionLab_NTroje_rub047_0027_circle_walk_poses', '0-KIT_183_walking_fast04_poses', '0-KIT_183_walking_fast08_poses',
    '0-KIT_9_WalkInClockwiseCircle10_poses', '0-ACCAD_Male2Running_c3d_C15 - run turn right 45_poses', '0-KIT_11_WalkInCounterClockwiseCircle01_poses',
    '0-KIT_425_walking_slow02_poses', '0-KIT_167_walking_slow04_poses', '0-KIT_348_walking_fast04_poses', '0-KIT_183_run02_poses', '0-KIT_9_walking_slow05_poses', '0-KIT_317_walking_fast09_poses', '0-KIT_183_walking_fast07_poses', '0-KIT_7_WalkingStraightForwards08_poses',
    '0-KIT_3_downstairs05_poses', '0-BioMotionLab_NTroje_rub073_0027_circle_walk_poses', '0-KIT_10_WalkInCounterClockwiseCircle03_poses'
    , '0-BioMotionLab_NTroje_rub062_0027_circle_walk_poses', '0-KIT_8_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_183_walking_slow04_poses', '0-KIT_424_walking_slow02_poses', '0-KIT_9_WalkInCounterClockwiseCircle08_poses',
    '0-KIT_3_walking_slow05_poses', '0-KIT_3_walking_slow06_poses', '0-KIT_348_walking_run10_poses', '0-KIT_513_upstairs_downstairs06_poses',
    '0-KIT_11_WalkInCounterClockwiseCircle07_poses', '0-KIT_424_downstairs01_poses', '0-KIT_6_WalkInClockwiseCircle01_1_poses',
    '0-ACCAD_Male2Walking_c3d_B23 -  side step right_poses', '0-KIT_348_walking_medium08_poses', '0-KIT_424_walking_run02_poses', '0-KIT_423_upstairs09_poses',
    '0-KIT_425_downstairs_07_poses', '0-KIT_359_walking_medium06_poses', '0-KIT_11_WalkInClockwiseCircle09_poses', '0-KIT_424_walking_slow01_poses',
    '0-Transitions_mocap_mazen_c3d_walksideways_turntwist180_poses', '0-KIT_9_WalkInClockwiseCircle08_poses', '0-KIT_317_walking_medium08_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle03_poses', '0-KIT_425_walking_slow04_poses', '0-KIT_183_run01_poses',
    '0-KIT_9_WalkInCounterClockwiseCircle01_poses', '0-KIT_205_walking_run09_poses', '0-ACCAD_MartialArtsWalksTurns_c3d_E19 - dodge left_poses',
    '0-KIT_348_walking_fast07_poses', '0-KIT_4_WalkInCounterClockwiseCircle08_poses', '0-KIT_8_WalkInClockwiseCircle08_poses',
    '0-BioMotionLab_NTroje_rub042_0027_circle_walk_poses', '0-KIT_3_upstairs06_poses', '0-KIT_12_WalkInCounterClockwiseCircle08_poses', '0-KIT_424_run04_poses',
    '0-KIT_9_WalkingStraightForwards03_poses', '0-KIT_3_walking_medium05_poses', '0-ACCAD_Male1Walking_c3d_Walk B10 - Walk turn left 45_poses',
    '0-KIT_425_walking_fast02_poses', '0-KIT_11_WalkInCounterClockwiseCircle04_poses', '0-KIT_9_walking_slow10_poses',
    '0-KIT_359_walking_slow10_poses', '0-KIT_348_walking_fast08_poses',
    '0-KIT_425_walking_slow10_poses', '0-KIT_314_walking_run10_poses', '0-KIT_183_walking_run08_poses', '0-KIT_317_walking_medium06_poses',
    '0-KIT_9_run03_poses', '0-KIT_6_WalkInCounterClockwiseCircle06_1_poses', '0-KIT_8_WalkInCounterClockwiseCircle07_poses', 
    '0-KIT_348_run01_poses',
    '0-KIT_348_walking_run05_poses', '0-KIT_8_WalkingStraightForwards04_poses', '0-KIT_314_walking_run03_poses',
    '0-KIT_167_walking_slow03_poses', '0-ACCAD_Male2Running_c3d_C11 - run turn left 90_poses', '0-KIT_317_walking_run03_poses', '0-KIT_3_upstairs03_poses',
    '0-ACCAD_Male1Running_c3d_Run C24 - quick side step left_poses', '0-KIT_3_walking_medium01_poses',
    '0-BioMotionLab_NTroje_rub098_0027_circle_walk_poses', '0-KIT_9_WalkInCounterClockwiseCircle02_poses', '0-KIT_424_walking_run09_poses',
    '0-KIT_167_walking_slow08_poses', '0-KIT_11_WalkingStraightForwards09_poses', '0-BioMotionLab_NTroje_rub041_0027_circle_walk_poses', '0-KIT_425_walking_03_poses',
    '0-KIT_314_walking_fast04_poses', '0-KIT_425_walking_medium01_poses', '0-KIT_167_upstairs01_poses', '0-KIT_167_walking_medium07_poses',
    '0-KIT_424_walking_slow06_poses', '0-TotalCapture_s1_walking1_poses', '0-KIT_424_walking_medium07_poses', '0-KIT_425_downstairs_02_poses',
    '0-BioMotionLab_NTroje_rub018_0027_circle_walk_poses', '0-KIT_205_walking_slow02_poses', '0-KIT_9_walking_slow08_poses',
    '0-KIT_359_walking_slow06_poses', '0-KIT_317_walking_medium04_poses', '0-KIT_205_walking_medium01_poses',
    '0-ACCAD_Male2Walking_c3d_B21 -  put down box to walk_poses', '0-BioMotionLab_NTroje_rub038_0027_circle_walk_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B16 - Walk turn change_poses', '0-KIT_425_walking_medium06_poses',
    '0-Eyes_Japan_Dataset_hamada_walk-22-look around-hamada_poses', '0-Eyes_Japan_Dataset_frederic_walk-04-fast-frederic_poses',
    '0-HumanEva_S2_Walking_1_poses', '0-KIT_9_walking_medium05_poses', '0-BioMotionLab_NTroje_rub017_0027_circle_walk_poses', '0-KIT_359_walking_medium04_poses',
    '0-KIT_425_walking_medium07_poses', '0-KIT_12_WalkingStraightForwards09_1_poses', '0-HumanEva_S2_Jog_1_poses', '0-KIT_7_WalkingStraightForwards09_poses',
    '0-KIT_9_walking_slow03_poses', '0-BioMotionLab_NTroje_rub064_0027_circle_walk_poses', '0-BioMotionLab_NTroje_rub085_0027_circle_walk_poses',
    '0-KIT_424_run02_poses', '0-KIT_317_walking_slow03_poses', '0-KIT_317_walking_medium07_poses', '0-KIT_8_WalkingStraightForwards10_poses',
    '0-BioMotionLab_NTroje_rub084_0027_circle_walk_poses', '0-BioMotionLab_NTroje_rub092_0023_circle_walk_poses',
    '0-KIT_8_WalkInClockwiseCircle06_poses', '0-KIT_359_walking_fast06_poses', '0-KIT_359_walking_fast02_poses',
    '0-ACCAD_Female1Running_c3d_C4 - Run to walk1_poses', '0-KIT_4_WalkingStraightForward01_poses', '0-KIT_3_upstairs_downstairs01_poses',
    '0-KIT_425_walking_medium02_poses', '0-KIT_9_walking_run09_poses', '0-KIT_424_upstairs01_poses', '0-KIT_8_WalkInCounterClockwiseCircle02_poses',
    '0-KIT_348_walking_medium10_poses', '0-KIT_317_walking_fast08_poses', '0-BioMotionLab_NTroje_rub060_0028_circle_walk2_poses',
    '0-KIT_424_downstairs08_poses', '0-KIT_348_walking_medium02_poses', '0-ACCAD_Female1Walking_c3d_B2 - walk to stand_poses',
    '0-KIT_348_run03_poses', '0-KIT_8_WalkInCounterClockwiseCircle03_poses', '0-KIT_317_walking_run08_poses', '0-BioMotionLab_NTroje_rub054_0027_circle_walk_poses',
    '0-KIT_314_walking_medium03_poses', '0-KIT_424_downstairs02_poses', '0-KIT_348_walking_medium03_poses',
    '0-KIT_359_walking_medium05_poses', '0-Eyes_Japan_Dataset_shiono_walk-10-shoulder bag-shiono_poses', '0-KIT_3_walking_fast08_poses', '0-KIT_7_WalkingStraightForwards06_poses',
    '0-KIT_424_walking_run01_poses', '0-KIT_9_walking_run02_poses',
    '0-BioMotionLab_NTroje_rub029_0027_circle_walk_poses', '0-KIT_3_walking_fast10_poses', '0-KIT_359_walking_slow07_poses'
]

amass_crawl_data = [
 '0-KIT_3_kneel_up_from_crawl05_poses',
 '0-KIT_3_kneel_down_to_crawl08_poses',
 '0-Transitions_mocap_mazen_c3d_crawl_runbackwards_poses',
 '0-ACCAD_Female1General_c3d_A10 - lie to crouch_poses',
 '0-BioMotionLab_NTroje_rub097_0020_lifting_heavy2_poses',
 '0-CMU_140_140_04_poses',
 '0-CMU_111_111_08_poses',
 '0-CMU_113_113_08_poses',
 '0-ACCAD_Male1General_c3d_General A10 -  Lie Down to Crouch_poses',
 '0-CMU_111_111_21_poses',
 '0-CMU_111_111_07_poses',
 '0-BMLmovi_Subject_11_F_MoSh_Subject_11_F_13_poses',
 '0-CMU_114_114_16_poses',
 '0-CMU_77_77_16_poses',
 '0-SFU_0017_0017_ParkourRoll001_poses',
 '0-ACCAD_Male1General_c3d_General A8 - Crouch to Lie Down_poses',
 '0-BMLmovi_Subject_5_F_MoSh_Subject_5_F_14_poses',
 '0-CMU_77_77_18_poses',
 '0-BMLmovi_Subject_89_F_MoSh_Subject_89_F_16_poses',
 '0-CMU_139_139_18_poses',
 '0-BioMotionLab_NTroje_rub075_0019_lifting_heavy1_poses',
 '0-BMLmovi_Subject_54_F_MoSh_Subject_54_F_12_poses',
 '0-BMLmovi_Subject_43_F_MoSh_Subject_43_F_6_poses',
 '0-CMU_140_140_01_poses',
 '0-CMU_140_140_02_poses',
 '0-Transitions_mocap_mazen_c3d_sit_jumpinplace_poses',
 '0-CMU_114_114_11_poses',
 '0-CMU_140_140_08_poses',
 '0-BMLmovi_Subject_35_F_MoSh_Subject_35_F_10_poses',
 '0-MPI_HDM05_mm_HDM_mm_03-03_01_120_poses',
 '0-BMLmovi_Subject_30_F_MoSh_Subject_30_F_6_poses',
 '0-BMLmovi_Subject_39_F_MoSh_Subject_39_F_16_poses'
]

amss_walk_data = ["0-KIT_359_walking_slow10_poses"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="amass_data/amass_db_smplh.pt")
    parser.add_argument("--out", type=str, default="smpl_data")
    args = parser.parse_args()
    
    amass_db = pickle.load(open(args.path,'rb'))
    
    random.shuffle(amass_run_data)
    random.shuffle(amss_walk_data)
    random.shuffle(amass_crawl_data)
    
    motion_types = {
        "run": amass_run_data, 
        "walk": amss_walk_data, 
        "crawl": amass_crawl_data
    }
    
    for motion_type, mocap_list in motion_types.items():
        dataset = {}
        
        for mocap_name in tqdm(mocap_list):
            start_name = mocap_name.split("-")[1]
            found = mocap_name in amass_db.keys()
            if not found:
                print(f"Not found!! {start_name}")
                continue
            dataset[mocap_name] = amass_db[mocap_name]
        
        if not osp.exists(args.out): os.mkdir(args.out)
        pickle.dump(dataset, open(f"{args.out}/amass_{motion_type}_train.pkl", 'wb'))