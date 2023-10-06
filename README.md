

# Environment Setup Guide
## introduction
This repository provides step-by-step instructions to help you set up a Linux-based environment from scratch. Follow the guide below to build your environment and run the server.

## Requirement
Before you start, ensure that you have the following requirements installed on your Linux system:

* Ubuntu 22.04.3
* Anaconda
* Python 3.7
* Kaldi
* Firebase

* Pre-trained model. You can download the pre-trained model at the following link [checklink](https://drive.google.com/drive/folders/1UThf-gQ4s2YkKf9aRGIHDdyfpT1ET_b7?usp=drive_link).
## Installation
## Kaldi
Follow the instructions provided in the [Kaldi installation guide](https://medium.com/@m.chellaa/install-kaldi-asr-on-ubuntu-830418a800b5)
 to install Kaldi on your system. Once the installation is complete, proceed with the following steps:
Change the file `path.sh` in your Kaldi installation path as required for your project.

```python
KALDI_ROOT=/home/moya/kaldi   # Change to your Kaldi installation path. 

. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
```

## Firebase
After setting up Firebase, you'll need to generate a new private key:

![Screenshot from 2023-09-20 17-04-05](https://github.com/Moyacaca/Speak-Slowly-Env/assets/117159970/dd282e39-b420-4f73-8229-f40f25da30c1)
1. Go to your Firebase project overview.
2. Navigate to Project Settings.
3. Select the Service Accounts tab.
4. Under the "Firebase Admin SDK" section, click the "Generate new private key" button to download the generated private key JSON file.
5. Place the private key JSON file into your project directory, typically named CNN-RNN-CTC.
Modify the path(line 17, line 26) in server.py to point to the location of the private key JSON file.

```python
cred = credentials.Certificate(
    "cgh-rcnn-flask-firebase-adminsdk-f92al-8d9038b979.json")  # Change to your private key!
save_path = 'run'
Path(save_path).mkdir(parents=True, exist_ok=True)
firebase_admin.initialize_app(
    cred, {'storageBucket': 'cgh-rcnn-flask.appspot.com'})   # connecting to firebase


def download_blob(bucket_name, filename, source_blob_name):
    credentials = service_account.Credentials.from_service_account_file(
        "cgh-rcnn-flask-firebase-adminsdk-f92al-8d9038b979.json")   # Change to your private key!
    storage_client = storage.Client(credentials=credentials)
```

## Running the Server
To run the server:

Verify that the server is configured to use port 5002 (you can change this in `server.py` if needed).
Open a terminal and navigate to your project directory.
Run the server using the following command:
```
python server.py
```
Your server should now be up and running.

## Usage
You can access your server at http://localhost:5002 (or the port you configured) in your web browser. Make sure you have the necessary dependencies installed in your Anaconda environment to use the server effectively.

Feel free to customize the server and environment to suit your specific project requirements.

Happy coding! 🚀
