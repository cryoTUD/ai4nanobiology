import os 
import dropbox

def upload_file_to_dropbox(filepath):
    ACCESS_TOKEN = "sl.u.AGbFJtTH3AqHPmNuTX82EuWsU7DQyD8eT1tpgexoZYoF8ypEjsB5ErpxCwGP1v7Ecw566LeCoWK2baKupsR8OwLV-MJ0nhiURqcj5Hw4ztrbUiyNPQymLBQweqHXZjpP84p4OCPtY03yTFg7walK7lwzKIliwG5dpFG0J9Ksu51e_i9TSkGrAdNBs11e0qzCEwnNL7tOl4-iIz0ofmS7RLc-TtAOYglHXx4GwSsg6WJEYfviqpGW0x-yrK3vjY_AggiMiYN2TwYN7UuNgaIJBzoMKB36SFjIHHqqAwNDwNsy7m_6VBUjJfACfJc7Nvt5ddMpkLnAkCvEutPVmf--ghJtSV92Pydmqr7abqamtoop1T3gjUz33x0-Ic52i-fYITft1NcRWTwxF8-3QTKrOX8ipyYgZRmiAhl32WehOKLY-wfdVnttubO3pLcFv5yFCdUPUWVjVOaL_FoYljwUZjp3uOgDVWurW-HQfSfEPKamm2Gu4U0X6_vX7sRwXnDKuxdKqzk9eNDxlYCfFtykrc9uaBAvasPj5oC4L4vVdebZtgzQtRmwb0Pcj2w0QW34EwmAy_QY4aApuKsYEHt2JBpKEYAH3C_Rz1Qioo_OfT-PVCG50BrfpmXNdFP6oaQU5knPTPI1zocucC1aoXA2fNE13mTHvEYIGZAaB1i3F2DzOXQ5twMESaH7AhL3twMT60SZt0TuXs3-OESgUFQV_SoqINPJZC7F4EgZAvep_gMJNT2NB9XfmYHH_C4ah0FQ_FyviiapAl_54O4XkAetGsXXwTr8oB_BFquwwtVUzxQIKADe5OYKvOuPtpJmYJ4-4n9aJp2ezUMplxQPo-LbTtod4TFAMoSote4boxfGgglHb_-dJ_LDdLzjJHNUWR-pocpFquO_MKEI7sDzzrZYVNuDk1pLMSRjptlOvBqpsGuJ8Ny3ocZrbNGMUCsLyQbGe4i9bysdEkPRlF6fcfB49YhbXEMfIhpSf8aCb-xjKRa6A2ZHKWKQBtV9LTHggm0uYqVFjYOZ8PCq3qTWEtrxnyt9XFSf4DbtG_XiCFg0rCxwZ16In1HvNfIbEPv-usWl2v2h6jX3lmCTVuH6HF77lWLMF_fc2ks-34QhR7PTxFU4Z30eMXfvsjlSiyjjXrRO4IgzLxQKmBCBXmPvlcdwZtdD365opN11bG767JEvNMAXcYZDCTWakvTp0qDyOETupkGq9jbL1fzBQiGQXWh8sdlwV0iRwwpmulJQx2-JNFxxWSdFJBrzmVME8eDd3jSFRCir4EaWaGgZtC2-TIWjkkOJ"
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    dropboxpath = "/App/NB4170_1"
    with open(filepath, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropboxpath,
            mode=dropbox.files.WriteMode.overwrite  # overwrite if exists
        )
    print(f"Uploaded: {filepath} → {dropboxpath}")
