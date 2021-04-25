from google_drive_downloader import GoogleDriveDownloader as gdd
import os

def main():

    if not os.path.exists('./processed_data'):
        print("downloading processed_data")
        gdd.download_file_from_google_drive(file_id='1NVFd0Lx_uSwOcNDhlh6FAwSnpP0yXt63', dest_path='./processed_data.zip', unzip=True)
        os.remove('./processed_data.zip')
    else:
        print("processed_data folder already exists!")


    if not os.path.exists('./saved_models'):
        print("downloading saved_models")
        gdd.download_file_from_google_drive(file_id='1VwW8-QYA8m1OghH4TU7g7A_INVbzzD2z', dest_path='./saved_models.zip', unzip=True)
        os.remove('./saved_models.zip')
    else:
        print("saved_models folder already exists!")


    if not os.path.exists('./output'):
        print("downloading output")
        gdd.download_file_from_google_drive(file_id='1IBqYk4RoZZSCgNQijQJG7kYA9mXzPlhr', dest_path='./output.zip', unzip=True)
        os.remove('./output.zip')
    else:
        print("output folder already exists!")

if __name__ == '__main__':
    main()