def create_example(xml_file):
    #process the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_name = root.find('filename').text
    file_name = image_name.encode('utf8')
    size=root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
        
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
        
    for member in root.findall('object'):
            
        classes_text.append('Person'.encode('utf8'))
        boundBox = member.find("bndbox")
        xmin.append(float(boundBox[0].text) / width)
        ymin.append(float(boundBox[1].text) / height)
        xmax.append(float(boundBox[2].text) / width)
        ymax.append(float(boundBox[3].text) / height)
        difficult_obj.append(0)
        #if you have more than one classes in dataset you can change the next line
        #to read the class from the xml file and change the class label into its 
        #corresponding integer number, u can use next function structure
        '''
        def class_text_to_int(row_label):
            if row_label == 'Person':
                return 1
            if row_label == 'car':
                return 2
            and so on.....
        '''
        classes.append(1)   # i wrote 1 because i have only one class(person)
        truncated.append(0)
        poses.append('Unspecified'.encode('utf8'))

    #read corresponding image
    full_path = os.path.join('../VOC2012/JPEGImages', '{}'.format(image_name))  #provide the path of images directory
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    #create TFRecord Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
    return example
