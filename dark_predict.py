import subprocess, os
from tqdm import tqdm

directory = '/datax/images/'

for profile in os.listdir(directory):
    print('PROCESSING:', profile)
    folder = directory + profile + '/'
    count = 1
    while count:
        if count > 10:
            print('SKIPPING:', profile)
            continue
        try:
            file = open('{}{}_detection.csv'.format(folder, profile), 'w+')
            count = 0
        except:
            print('TRY #{}:'.format(count), profile)
            count += 1
            continue
    for filename in tqdm(os.listdir(folder)):
        try:
            if filename.endswith('.jpg'):
                command = './darknet detect cfg/yolov3.cfg cfg/yolov3.weights {}{} -thresh 0.5'.format(folder, filename)
                subprocess.check_call(command.split())
                objects = open('prediction_details.txt', 'r').read().strip('\n').split('\n')
                file.write(filename)
                for obj in objects:
                    (label, conf, left, right, top, bot) = obj.split(',')
                    file.write(',"{},{},{},{},{},{}"'.format(label, conf, left, right, top, bot))
                file.write('\n')
        except:
            print('ERROR. SKIPPING:', profile, filename)
            continue
