#!/usr/bin/env python
from pathlib import Path

from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    if not Path("models/dronet_dcgan_generator.onnx").exists():
        print(
            "downloading DroNet DCGAN generator to models/dronet_dcgan_generator.onnx"
        )
        gdd.download_file_from_google_drive(
            file_id="1shlyt_LGP3C6V_MX-A0wf9VtuPHyWlA2",
            dest_path="models/dronet_dcgan_generator.onnx",
        )

    if not Path("models/dronet_fc_vae_decoder.onnx").exists():
        print("downloading DroNet FC-VAE decoder to models/dronet_fc_vae_decoder.onnx")
        gdd.download_file_from_google_drive(
            file_id="1SSh_MrqO3hRJyO4ak8hgLGwNiYgKh5Nk",
            dest_path="models/dronet_fc_vae_decoder.onnx",
        )

    if not Path("models/dronet_fc_vae.onnx").exists():
        print("downloading DroNet FC-VAE to models/dronet_fc_vae.onnx")
        gdd.download_file_from_google_drive(
            file_id="1LFEsA7uBsisH9V1Px6r0x-q_Jb341Eul",
            dest_path="models/dronet_fc_vae.onnx",
        )

    if not Path("models/dronet_vae_vae_encoder.onnx").exists():
        print(
            "downloading DroNet Conv-VAE encoder to models/dronet_vae_vae_encoder.onnx"
        )
        gdd.download_file_from_google_drive(
            file_id="1V7Zan0pFe30Dsl0Z2RQvHjJskD2eIixu",
            dest_path="models/dronet_vae_vae_encoder.onnx",
        )

    if not Path("models/dronet_vae_vae.onnx").exists():
        print("downloading DroNet Conv-VAE to models/dronet_vae_vae.onnx")
        gdd.download_file_from_google_drive(
            file_id="1f8vfj4YIOr_xONczwgWWyS_1_NHDw09e",
            dest_path="models/dronet_vae_vae.onnx",
        )

    if not Path("logs/falsification_logs").exists():
        print("downloading logs to logs/")
        gdd.download_file_from_google_drive(
            file_id="1mnfJFtAeo5yURJnkwAGDH_kkATxj88_U",
            dest_path="./logs.zip",
            unzip=True,
        )
        Path("logs.zip").unlink()


if __name__ == "__main__":
    main()
