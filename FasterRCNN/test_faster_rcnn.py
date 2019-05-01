# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import os
import numpy as np
import cntk
import sys
import cv2
from cntk import Trainer, load_model, Axis, input_variable, parameter
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results
def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN_config import cfg as detector_cfg
    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.MycombinedGrocery_config import cfg as dataset_cfg
    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    cfg = get_configuration()

    prepare(cfg, False)
    #cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))
    # train and test
    #trained_model = train_faster_rcnn(cfg)

    #image_path = '/home/analyticscrack/cntk/cntk/Examples/Image/DataSets/MyDataSet/testImages/ch01_20180328115228_2567.jpg'

    print(cfg["DATA"].CLASSES)
    print(cfg["DATA"].TEST_MAP_FILE)

    print(cfg.RESULTS_NMS_THRESHOLD)
    print(cfg.RESULTS_NMS_CONF_THRESHOLD)


    model_path = '/home/nouveau-labs/cntk/Examples/Image/Detection/FasterRCNN/Output/faster_rcnn_eval_AlexNet_e2e.model' 


    eval_model = load_model(model_path)

    evaluator = FasterRCNN_Evaluator(eval_model, cfg)
    print(cfg["DATA"].CLASSES)

    #regressed_rois, cls_probs = evaluator.process_image('/home/analyticscrack/cntk/cntk/Examples/Image/DataSets/MyDataSet/testImages/ch01_20180328115228_2567.jpg')
    num_eval = 100 
    results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
    plot_test_set_results(evaluator, num_eval, results_folder, cfg)

    
    print(cfg["DATA"].TEST_MAP_FILE)
    # write AP results to output

    #for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    #print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))
    # Plot results on test set images
    #if cfg.VISUALIZE_RESULTS:
    #    num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
    #    results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
    #    evaluator = FasterRCNN_Evaluator(trained_model, cfg)
    #    plot_test_set_results(evaluator, num_eval, results_folder, cfg)
    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)
