import os
from colorama import Fore
from predictions import predict


# load images to predict from paths
#               ....                       /    elbow1.jpg
#               Hand          fractured  --   elbow2.png
#           /                /             \    .....
#   test   -   Elbow  ------
#           \                \         /        elbow1.png
#               Shoulder        normal --       elbow2.jpg
#               ....                   \
#
def load_path(path):
    dataset = []
    for body in os.listdir(path):
        body_part = body
        path_p = path + '/' + str(body)
        for lab in os.listdir(path_p):
            label = lab
            path_l = path_p + '/' + str(lab)
            for img in os.listdir(path_l):
                if img.startswith("cam_") or img.startswith("gradcam_"):
                    continue
                img_path = path_l + '/' + str(img)
                dataset.append(
                    {
                        'body_part': body_part,
                        'label': label,
                        'image_path': img_path,
                        'image_name': img
                    }
                )
    return dataset


categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']


def reportPredict(dataset):
    total_count = 0
    part_count = 0
    status_count = 0

    print(Fore.YELLOW +
          '{0: <28}'.format('Name') +
          '{0: <14}'.format('Part') +
          '{0: <20}'.format('Predicted Part') +
          '{0: <20}'.format('Status') +
          '{0: <20}'.format('Predicted Status'))
    for img in dataset:
        body_part_predict = predict(img['image_path'])
        result_data = predict(img['image_path'], body_part_predict)
        
        # Handle dictionary return from safety update
        if isinstance(result_data, dict):
            predicted_status_str = result_data['result'] # DETECTED / UNCERTAIN
            # Use original_result for binary accuracy tracking
            pred_label = result_data['original_result']
        else:
            predicted_status_str = str(result_data)
            pred_label = result_data

        if img['body_part'] == body_part_predict:
            part_count = part_count + 1
        if img['label'] == pred_label:
            status_count = status_count + 1
            color = Fore.GREEN
        else:
            color = Fore.RED
        print(color +
              '{0: <28}'.format(img['image_name']) +
              '{0: <14}'.format(img['body_part']) +
              '{0: <20}'.format(body_part_predict) +
              '{0: <20}'.format((img['label'])) +
              '{0: <20}'.format(predicted_status_str))

    print(Fore.BLUE + '\npart acc: ' + str("%.2f" % (part_count / len(dataset) * 100)) + '%')
    print(Fore.BLUE + 'status acc: ' + str("%.2f" % (status_count / len(dataset) * 100)) + '%')
    return


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
test_dir = THIS_FOLDER + '/test/'
reportPredict(load_path(test_dir))
