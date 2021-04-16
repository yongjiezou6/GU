CHAR_VECTOR = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
letters = [letter for letter in CHAR_VECTOR]
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

# network parameters
letter_num = len(letters)
province_num = len(provinces)

img_w, img_h = 152,56
lp_text_len = 7

# training parameters
use_gpu = '0'
gpu_num = len(use_gpu.split(','))

train_file_path ='/home/zouyongjie/lpdr/data/ccpd/train/lp_train'
valid_file_path ='/home/zouyongjie/lpdr/data/ccpd/val/lp_val'


# names = 'challenge'
# test_file_path = '/home/work/zyj/yongjie/datasets/home/booy/booy/ccpd_dataset/lp_ccpd_old_data/{}/lp_{}'.format(names,names)
# test_file_path = '/home/work/zyj/yongjie/datasets/home/booy/booy/ccpd_dataset/tt'#'/home/work/zyj/yongjie/datasets/CLPD/lp_CLPD'

batch_size = 32
val_batch_size = 32
epochs = 200


#result
plt_model_path = 'logs'


#test paraments
CUDA_VISIBLE_DEVICES = '-1'
pre_save_path = test_file_path.split('/')[-1]
weight_path ='logs/ILPNET.h5'
error_path = plt_model_path +'/'+test_file_path.split('/')[-1]+'_error'
correct_path = plt_model_path +'/'+test_file_path.split('/')[-1]+'_correct'




