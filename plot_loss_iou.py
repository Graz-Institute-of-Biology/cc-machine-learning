import pandas as pd

import matplotlib.pyplot as plt

notes = 'Background problems'

base_path = r'C:\Users\faulhamm\Documents\Philipp\Code\cc-machine-learning\results\01-cc-atto\exp'
exp_name_bkg_ok = 'exp_ATTO_mit_b1_s10_cv0_dfc8c2'
exp_name_bkg_notok = 'exp_ATTO_mit_b1_s10_cv0_2a44cf'


train_log = 'train_log.csv'
valid_log = 'valid_log.csv'

train_file_path = f'{base_path}/{exp_name_bkg_notok}/{train_log}'
valid_file_path = f'{base_path}/{exp_name_bkg_notok}/{valid_log}'


# Read CSV file
train_df = pd.read_csv(train_file_path)
valid_df = pd.read_csv(valid_file_path)

train_loss = train_df['Focal_loss']
train_iou = train_df['iou_score']*100

valid_loss = valid_df['Focal_loss']
valid_iou = valid_df['iou_score']*100

# Plot all columns
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(valid_loss, label='Valid Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_iou, label='Train IoU')
plt.plot(valid_iou, label='Valid IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.ylim(0,100)
plt.legend()
plt.grid()

plt.suptitle(f'Loss and IoU for {exp_name_bkg_notok} \n {notes}', fontsize=16)
plt.tight_layout()
plt.show()
