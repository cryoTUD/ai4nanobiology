import os 
import dropbox

def upload_file_to_dropbox(filepath):
    ACCESS_TOKEN = "sl.u.AGZaQfhGyxuZszDVaMOpoZhYdZZ3Mxx4A3yTNt8iMVNrCMN7RKd0dto9nDnRSNjm7oetwGF7WmkxW1gNoq-4GH7PJYo5094LNZYIJ91KQ94ROfy7j7LejCuqK1cm2cGZm-9zFT2F4WOw04YeJbp4K9aKYsf-jBdFkM3d3AKUnaDrjCS0J1YIwOVaTicmTycX1tpE6lkdSAw4fTAe3Vgt6hAbISW9sozLWQuX1j_pphfDH3ze-VWUWs9wCdTgIvujtau3iw6sTEN_YxYGyXNj33VSD8c2vhMjg4UFnOKQsKWxgvmlBz46ISfyFWKFDT3EKlSkjhpa1tSbHRpHQtQAhxckkbEm-1PHAngcQRVoVbollR-xrku0oHMjRh8WCzzCLqz2Fwh50pfkkQOShXyat5XBdcdAXQOSRP2rw-OPkVQlg_pjSDkPV_P_D79ui7do7zGTAcr2_3ufda3Ybr2_ySYD72njyaA0w7eVaKkHzZQgLmqrESYmMeL8nc37GZwtaSTpgpstrnbNNBGUIIKnp8LwNgV9SG-2lAu5j_glGtlgsmbSAV-73h2s8zPHtD_c_cGtdGvS8LEp3weVNvNVY3Q-8ChFFWGlVPPoX2sk9dMtzPYVhRgKfALPWzxkucU8ri9EuAtjLrpj1Ax24garUEeAywm-2m0jyosRRn-SBy1SV3venyorfzSXklkDySON_9bVzmQrhv3Jf3pBn9eeapNB3WukuPN1lNR63x5wkoBI16x2v89iu6r0Xs1yaRYTfDRrlEZp_uoigzRPVUKdM8MImhd128Xd6cOHsaD5DSemR2cRXYEi4QrZUCcnI1DAm6ETrCHetaP4kpg8scEd_Avk3OHsObQdq-R6OcpvlD-EadYA7tNxXYliFjESX175xeLXlITnYhI4zDoieivp5_1UgOx6nzgCHthua120bx5njB7ffqrG8UQzvZeK_wobjvvtxgB1gr5joT0eUPo4me2OjsawEM9tM-Jxjh049K39dBXk2CUxHaUNpEwD4v_-JhKhTAmdAJl6Q9L-oXlPsLxbi3MCUXav0xiHb_o7ekBGTIO0pK1WvFcaVJcZ0YmSYh--cZBBDSJ7laUXtZsoqC5xdpei0XnaVxcgWz4KIq-NX-jlpU7RfqSgMbU5FdrdfPi7WjTgHwz2B9nVI1yV6KZOAh1qoAd6aTvfdiweQhc_UF_BEOnuTfmtslc7Zm1qJUajMZ4FqszZTr-qPuiwB1529HhYw6_K_viRQ449yROs0IiQTSHavuOFhP3ceeSskcgJ6F-zqERy-pVczjqXL7IZ"
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    dropboxpath = "/App/NB4170_1"
    with open(filepath, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropboxpath,
            mode=dropbox.files.WriteMode.overwrite  # overwrite if exists
        )
    print(f"Uploaded: {filepath} → {dropboxpath}")
