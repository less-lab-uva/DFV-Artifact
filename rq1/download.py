from google_drive_downloader import GoogleDriveDownloader as gdd
import os

def main():

    if not os.path.exists('./processed_data'):
        print("downloading processed_data")
        gdd.download_file_from_google_drive(file_id='1K7f0JHL1oYHcAVkVixcC7TSDsfBT6R00', dest_path='./processed_data.zip', unzip=True)
        os.remove('./processed_data.zip')
    else:
        print("processed_data folder already exists!")


    if not os.path.exists('./saved_models'):
        print("downloading saved_models")
        gdd.download_file_from_google_drive(file_id='1kGpMObeJ3LvA1-ra0NgLJjc1fFVSaPod', dest_path='./saved_models.zip', unzip=True)
        os.remove('./saved_models.zip')
    else:
        print("saved_models folder already exists!")


    if not os.path.exists('./output'):
        print("downloading output")
        gdd.download_file_from_google_drive(file_id='1YrtybgXlblVrcXMJv9--jV-GfNStnxZu', dest_path='./output.zip', unzip=True)
        os.remove('./output.zip')
    else:
        print("output folder already exists!")

if __name__ == '__main__':
    main()