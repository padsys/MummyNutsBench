
# Copyright (c) 2023, University of California, Merced. All rights reserved.

# This file is part of the MummyNutsBench software package developed by
# the team members of Prof. Xiaoyi Lu's group (PADSYS Lab) at the University
# of California, Merced.

# For detailed copyright and licensing information, please refer to the license
# file LICENSE in the top level directory.



# run conda activate mnut first
# conda create --name ssdmobile python=3.7.13

import os
import sys

WORK_DIR = os.getcwd()
ENABLE_CUDA = True

class SSD:
    CUSTOM_MODEL_NAME = 'mNuts_centerNet' 
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'https://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'nuts_label_map.pbtxt'

    paths = {
        'ssdmobilenet_PATH': os.path.join('Tensorflow', 'ssdmobilenet'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'ssdmobilenet','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'ssdmobilenet','images'),
        'TRAIN_PATH': os.path.join('Tensorflow', 'ssdmobilenet','images', 'train'),
        'TEST_PATH': os.path.join('Tensorflow', 'ssdmobilenet','images', 'test'),
        'VALID_PATH': os.path.join('Tensorflow', 'ssdmobilenet','images', 'valid'),
        'MODEL_PATH': os.path.join('Tensorflow', 'ssdmobilenet','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'ssdmobilenet','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'ssdmobilenet','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'ssdmobilenet','models',CUSTOM_MODEL_NAME, 'export'),
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
    }

    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'ssdmobilenet','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    def setup():
        #Create dirs
        for path in SSD.paths.values():
            if not os.path.exists(path):
                if os.name == 'posix':
                    os.system(f"mkdir -p {path}")
                if os.name == 'nt':
                    os.system(f"mkdir {path}")
            
        if not os.path.exists("bin"):
            os.system(f"mkdir bin")

        # I think this link is okay to be public
        os.system('curl -L "https://app.roboflow.com/ds/XVJeUDRiuq?key=1R4xISrJZj" > roboflow.zip; unzip roboflow.zip -d Tensorflow/ssdmobilenet/images/; rm roboflow.zip')
        os.system('cp Tensorflow/ssdmobilenet/images/train/nuts_label_map.pbtxt Tensorflow/ssdmobilenet/annotations/')

        if not os.path.exists(os.path.join(SSD.paths['APIMODEL_PATH'], 'research', 'object_detection')):
            os.system(f"git clone https://github.com/tensorflow/models {SSD.paths['APIMODEL_PATH']}")
        
        # dl protoc
        os.system("wget -O bin/protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v21.1/protoc-21.1-linux-x86_64.zip")
        os.system("mkdir -p protoc && unzip -d bin/protoc bin/protoc.zip")
        os.environ["PATH"] = f"{os.environ['PATH']}:{WORK_DIR}/bin/protoc/bin"

        # need to run twice to fix dependencies
        os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python3 -m pip install . ")
        os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python3 -m pip install . ")

        # dl pretrained models
        os.system(f"wget --no-check-certificate {SSD.PRETRAINED_MODEL_URL}")
        os.system(f"mv {SSD.PRETRAINED_MODEL_NAME+'.tar.gz'} {SSD.paths['PRETRAINED_MODEL_PATH']}")
        os.system(f"cd {SSD.paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {SSD.PRETRAINED_MODEL_NAME+'.tar.gz'}")
        os.system(f"cp {os.path.join(SSD.paths['PRETRAINED_MODEL_PATH'], SSD.PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(SSD.paths['CHECKPOINT_PATH'])}")

        # fix "ImportError: cannot import name 'builder' from 'google.protobuf.internal'"
        os.system("pip uninstall protobuf; pip install protobuf==3.20.0")

        if ENABLE_CUDA:
            # need to change this to dl the zip and extract the libraries
            # os.system("apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2")
            os.environ["LD_LIBRARY_PATH"] = '/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4/targets/x86_64-linux/lib/:/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/math_libs/11.4/targets/x86_64-linux/lib/:/home/andrew/centernet/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib'
            os.environ["CUDA_DIR"] = '/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4'

        import object_detection
        import tensorflow as tf
        from object_detection.utils import config_util
        from object_detection.protos import pipeline_pb2
        from google.protobuf import text_format

        #Setup pipeline.config
        setBatchSize = '8'
        pipeline = ''
        num_classes = '1'
        with open(os.path.join(SSD.paths['CHECKPOINT_PATH'], 'pipeline.config')) as file:
            pipeline = file.read()
            pipeline = pipeline.replace("PATH_TO_BE_CONFIGURED", 'Tensorflow/ssdmobilenet/pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0', 1)
            pipeline = pipeline.replace('PATH_TO_BE_CONFIGURED', 'Tensorflow/ssdmobilenet/images/train/nuts_label_map.pbtxt', 1)
            pipeline = pipeline.replace('PATH_TO_BE_CONFIGURED', 'Tensorflow/ssdmobilenet/images/train/nuts.tfrecord', 1)
            pipeline = pipeline.replace('PATH_TO_BE_CONFIGURED', 'Tensorflow/ssdmobilenet/images/valid/nuts_label_map.pbtxt', 1)
            pipeline = pipeline.replace('PATH_TO_BE_CONFIGURED', 'Tensorflow/ssdmobilenet/images/valid/nuts.tfrecord', 1)
            pipeline = pipeline.replace('batch_size: 128', 'batch_size: '+setBatchSize, 1)
            pipeline = pipeline.replace("fine_tune_checkpoint_type: \"classification\"", "fine_tune_checkpoint_type: \"detection\"", 1)
            pipeline = pipeline.replace("num_classes: 90", "num_classes: 1", 1)

        with open(os.path.join(SSD.paths['CHECKPOINT_PATH'], 'pipeline.config'), 'w') as file:
            file.write(pipeline)
        
    def run():
        if ENABLE_CUDA:
            # need to change this to dl the zip and extract the libraries
            # os.system("apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2")
            os.environ["LD_LIBRARY_PATH"] = '/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4/targets/x86_64-linux/lib/:/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/math_libs/11.4/targets/x86_64-linux/lib/:/home/andrew/centernet/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib'
            os.environ["CUDA_DIR"] = '/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4'
            os.environ["XLA_FLAGS"] = '--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4'

        TRAINING_SCRIPT = os.path.join(SSD.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        command = "python3 {} --model_dir={} --pipeline_config_path={} --num_train_steps=3000".format(TRAINING_SCRIPT, SSD.paths['CHECKPOINT_PATH'], SSD.files['PIPELINE_CONFIG'])
        print(command)

        os.system(command)

if sys.argv[1] == "setup":
    SSD.setup()
elif sys.argv[1] == "run":
    SSD.run()
