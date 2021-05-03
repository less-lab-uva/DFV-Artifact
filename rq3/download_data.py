#!/usr/bin/env python
from pathlib import Path

from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    if not Path("Data/dronet.200").exists():
        print("downloading processed data to Data/")
        gdd.download_file_from_google_drive(
            file_id="1z-VcbR9ztYNXMiwYwnQ-RXfTJI4bsSBp",
            dest_path="Data/dronet_dataset.zip",
            unzip=True,
        )
        Path("Data/dronet_dataset.zip").unlink()


if __name__ == "__main__":
    main()
