#89.35(accuracy)


import math
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import argparse
from utils import load_data
import time
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ferPlus_path', type=str, default='../DB/FER+/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.15, help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2, help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()
    

class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 8, drop_rate = 0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        
        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out
        
def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    # if isinstance(m, CondConv2d):
        # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # init_weight_fn = get_condconv_initializer(
            # lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        # init_weight_fn(m.weight)
        # if m.bias is not None:
            # m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()
        
def run_training():

    # Running device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    
    print("Started Training")
    args = parse_args()
    imagenet_pretrained = True
    res18 = Res18Feature(pretrained = imagenet_pretrained, drop_rate = args.drop_rate) 
    
    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)
            
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict = False)
    
    res18 = res18.to(device)
    
    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=30,
                                                translate=(.1, .1),
                                                scale=(1.0, 1.25),
                                                interpolation=transforms.InterpolationMode.BILINEAR)]
            
    train_dataset = load_data.FERplus(idx_set=0,
                                    max_loaded_images_per_label=5000,
                                    transforms=transforms.Compose(data_transforms),
                                    base_path_to_FER_plus=args.ferPlus_path)
    print('Train set size:', train_dataset.__len__())
        
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = args.batch_size,
                                                num_workers = args.workers,
                                                shuffle = True,  
                                                pin_memory = True)
    
    val_dataset = load_data.FERplus(idx_set=1,
                                max_loaded_images_per_label=100000,
                                transforms=None,
                                base_path_to_FER_plus=args.ferPlus_path)
    print('Validation set size:', val_dataset.__len__())
        
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size = args.batch_size,
                                            num_workers = args.workers,
                                            shuffle = False,  
                                            pin_memory = True)
        
    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                        momentum=args.momentum,
                                        weight_decay = 1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()
        
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
        
    print("Starting epochs:")
    for i in range(1, args.epochs + 1):
        start = time.time()
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0) 
            iter_cnt += 1
            tops = int(batch_sz* beta)
            optimizer.zero_grad()
            imgs = imgs.to(device)
            attention_weights, outputs = res18(imgs)
                
                # Rank Regularization
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest = False)
    
            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            # diff  = margin_1 - (high_mean - low_mean)
            diff  = low_mean - high_mean + margin_1
    
            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0
    
            targets = targets.to(device)
            loss = criterion(outputs, targets) + RR_loss 
            loss.backward()
            optimizer.step()
                
            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
    
            # Relabel samples
            if i >= args.relabel_epoch:
                sm = torch.softmax(outputs, dim = 1)
                Pmax, predicted_labels = torch.max(sm, 1) # predictions
                Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                true_or_false = Pmax - Pgt > margin_2
                update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                label_idx = indexes[update_idx] # get samples' index in train_loader
                relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
                train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader                    
        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
            
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            res18.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                _, outputs = res18(imgs.to(device))
                targets = targets.to(device)
                loss = criterion(outputs, targets)
                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(outputs, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)
                    
            running_loss = running_loss/iter_cnt   
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))
        
        end = time.time() 
        print(f'[*] epoch time = {end - start:.2f}s | ')
        """if acc > 0.87 :
                torch.save({'iter': i,
                            'model_state_dict': res18.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('SCN/models', "epoch"+str(i)+"_acc"+str(acc)+".pth"))
                print('Model saved.')"""
            
if __name__ == "__main__": 
    start = time.time()                   
    run_training()
    end = time.time()
    print(f'[!] total time = {timedelta(seconds=end - start)}s')