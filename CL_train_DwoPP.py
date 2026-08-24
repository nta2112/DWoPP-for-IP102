'''
DwoPP for IP102 - Continual Meta Metric Learning for Lifelong Object Re-Identification
Adapted for IP102 dataset with 25 classes (7/6/6/6 task split)
'''
import copy
import torch
import torch.optim as optim
from torch.nn import DataParallel
import time
import numpy as np
import os
import random
from model import resnet_model
from data.CL_loader import make_dataloader
from loss import make_loss
from config import get_parser
import torch.nn.functional as F
import utils
from metrics import evaluate_retrieval, MetricsLogger
from utils_extra import unwrap_model, get_gpu_ids, auto_detect_gpus, find_data_root, find_file_deep


def init_seed(args, gids):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if gids is not None:
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_model(args, gids=None):
    model = resnet_model(remove_downsample=args.remove_downsample)
    return model


def adjust_lr_exp(optimizer, base_lr, epoch, num_epochs, decay_start_epoch):
    if epoch < decay_start_epoch:
        return
    for g in optimizer.param_groups:
        g['lr'] = base_lr * (0.005 ** (float(epoch + 1 - decay_start_epoch)
                                        / (num_epochs + 1 - decay_start_epoch)))
    print('=====> lr adjusted to {:.9f}'.format(g['lr']).rstrip('0'))


def extract_features(model, dataloader, gids=None, normalize=True):
    model.eval()
    all_feats = []
    all_labels = []
    all_ids = []
    with torch.no_grad():
        for images, labels in dataloader:
            if gids is not None:
                images = images.cuda(gids[0])
            feats = model(images)
            if normalize:
                feats = F.normalize(feats, p=2, dim=1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            if hasattr(dataloader.dataset, 'img_ids'):
                batch_ids = [dataloader.dataset.img_ids[i] for i in range(len(all_feats[-1]))]
                all_ids.extend(batch_ids)
    if all_ids:
        return torch.cat(all_feats), torch.cat(all_labels), np.array(all_ids)
    return torch.cat(all_feats), torch.cat(all_labels), None


def build_eval_loaders(args, task_id, seen_classes, gids=None):
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from data.ip102 import IP102
    
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gallery_set = IP102(args.dataset_root, eval_transform, split='val',
                         filtered_class_path=args.filtered_class_path,
                         classes_txt_path=args.classes_txt_path)
    query_set = IP102(args.dataset_root, eval_transform, split='test',
                       filtered_class_path=args.filtered_class_path,
                       classes_txt_path=args.classes_txt_path)
    
    batch_size = 32
    if gids and len(gids) > 1:
        batch_size = batch_size * len(gids)
    
    gallery_loader = DataLoader(gallery_set, batch_size=batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False)
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)
    
    return gallery_loader, query_loader, gallery_set, query_set


def train(args, model, optimizer, criterion, task_id, gids=None, old_model=None, logger=None, all_task_maps=None):

    model.train()
    t0 = int(time.time())

    for epoch in range(args.num_epochs):
        train_loss = []
        dmml_losses = []
        know_distill_losses = []

        if epoch % 10 == 0:
            dataloader = make_dataloader(args, task_id, epoch)

        print('=== Epoch {}/{} ==='.format(epoch, args.num_epochs))
        if args.num_epochs > 1:
            adjust_lr_exp(optimizer, args.lr, epoch, args.num_epochs, args.lr_decay_start_epoch)
        
        for iteration, (image, label) in enumerate(dataloader):
            if args.cuda:
                image, label = image.cuda(gids[0]), label.cuda(gids[0])
            feat = model(image)

            dmml_loss = criterion(feat, label)
            
            # Check for NaN
            if torch.isnan(dmml_loss):
                print(f'WARNING: NaN dmml_loss detected at iteration {iteration}')
                continue

            if task_id > 0 and args.weight_knowledge_distill > 0:
                feat_old_model = old_model(image)

                reshape_feat_new_model = feat.reshape(-1, args.num_support + args.num_query, 2048)
                enc_data_query = reshape_feat_new_model[:, args.num_support:, :].squeeze(1)

                if args.distillation_dist_metric == 'euclidean':
                    enc_proto = reshape_feat_new_model[:, :args.num_support, :].mean(1)
                    mix_task_new_logits = utils.decode(enc_proto, enc_data_query)
                elif args.distillation_dist_metric == 'cosine':
                    enc_proto = F.normalize(reshape_feat_new_model[:, :args.num_support, :]).mean(1)
                    mix_task_new_logits = utils.cosine_decode(enc_proto, enc_data_query)
                else:
                    raise NotImplementedError
                
                if args.remove_positive_pair:
                    identity = torch.eye(len(mix_task_new_logits))
                    mix_task_new_logits = mix_task_new_logits[(1-identity).bool()]
                    mix_task_new_logits = mix_task_new_logits.reshape(len(identity), -1)

                mix_task_new_logits = F.softmax(mix_task_new_logits, dim=1)

                if args.temperature != 1.0:
                    eps = 1e-5
                    T = args.temperature
                    mix_task_new_logits = F.normalize(mix_task_new_logits.pow(1/T), dim=1, p=1)
                    mix_task_new_logits = F.normalize(mix_task_new_logits + eps / mix_task_new_logits.size(1), dim=1, p=1)

                reshape_feat_old_model = feat_old_model.reshape(-1, args.num_support + args.num_query, 2048)
                enc_data_query = reshape_feat_old_model[:, args.num_support:, :].squeeze(1)
                if args.distillation_dist_metric == 'euclidean':
                    enc_proto = reshape_feat_old_model[:, :args.num_support, :].mean(1)
                    mix_task_old_logits = utils.decode(enc_proto, enc_data_query)
                elif args.distillation_dist_metric == 'cosine':
                    enc_proto = F.normalize(reshape_feat_old_model[:, :args.num_support, :]).mean(1)
                    mix_task_old_logits = utils.cosine_decode(enc_proto, enc_data_query)
                else:
                    raise NotImplementedError
                
                if args.remove_positive_pair:
                    identity = torch.eye(len(mix_task_old_logits))
                    mix_task_old_logits = mix_task_old_logits[(1-identity).bool()]
                    mix_task_old_logits = mix_task_old_logits.reshape(len(identity), -1)

                mix_task_old_logits = F.softmax(mix_task_old_logits, dim=1)
                
                if args.temperature != 1.0:
                    eps = 1e-5
                    T = args.temperature
                    mix_task_old_logits = F.normalize(mix_task_old_logits.pow(1/T), dim=1, p=1)
                    mix_task_old_logits = F.normalize(mix_task_old_logits + eps / mix_task_old_logits.size(1), dim=1, p=1)

                kl_div_mix_task = (mix_task_old_logits.clamp(min=1e-4) * (mix_task_old_logits.clamp(min=1e-4)
                                / mix_task_new_logits.clamp(min=1e-4)).log()).sum() / len(mix_task_old_logits)
                kl_div_mix_task = kl_div_mix_task * args.weight_knowledge_distill
                
                # Check for NaN in distillation loss
                if torch.isnan(kl_div_mix_task):
                    print(f'WARNING: NaN kl_div_mix_task detected at iteration {iteration}')
                    kl_div_mix_task = torch.tensor(0.0).cuda(gids[0]) if gids else torch.tensor(0.0)
            else:
                kl_div_mix_task = torch.tensor(0.0).cuda(gids[0]) if gids else torch.tensor(0.0)

            loss = dmml_loss + kl_div_mix_task
            
            # Check for NaN in total loss
            if torch.isnan(loss):
                print(f'WARNING: NaN total loss detected at iteration {iteration}')
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss.append(loss.item())
            dmml_losses.append(dmml_loss.item())
            know_distill_losses.append(kl_div_mix_task.item())

            print('Episode: {}, Loss: {:.6f}, dmml_loss: {:.6f}, kl_div_mix_task: {:.6f} '
                  .format(iteration, loss.item(), dmml_loss.item(), kl_div_mix_task.item()))

        avg_training_loss = np.mean(train_loss)
        avg_dmml_losses = np.mean(dmml_losses)
        avg_know_distill_losses = np.mean(know_distill_losses)

        print('Average loss: {:.6f}, dmml_losses: {:.6f}, know_distill_losses: {:.6f}'
              .format(avg_training_loss, avg_dmml_losses, avg_know_distill_losses))
        t = int(time.time())
        print('Time elapsed: {}h {}m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))

    model_save_path = os.path.join(args.exp_root, args.method, 
                                '{}_model_last_task_{}.pth'.format(args.method, task_id))
    model_to_save = unwrap_model(model)
    torch.save(model_to_save.state_dict(), model_save_path)
    print('Final model saved.')

    seen_classes = []
    for t in range(task_id + 1):
        seen_classes.extend(args.task_classes[t])
    
    gallery_loader, query_loader, gallery_set, query_set = build_eval_loaders(args, task_id, seen_classes, gids)
    gallery_feats, gallery_labels, gallery_ids = extract_features(model, gallery_loader, gids)
    query_feats, query_labels, query_ids = extract_features(model, query_loader, gids)

    metrics = evaluate_retrieval(query_feats, gallery_feats, query_labels, gallery_labels,
                                  seen_classes=seen_classes,
                                  query_ids=query_ids, gallery_ids=gallery_ids)
    
    if all_task_maps is not None:
        all_task_maps.append([metrics['mAP']])
    
    if logger:
        plasticity, forgetting, overall = logger.log(task_id, len(seen_classes), metrics, all_task_maps)
        print(f'Lifelong: Plasticity={plasticity:.4f}, Forgetting={forgetting:.4f}, Overall={overall:.4f}')

    print('Retrieval: R@1={:.4f}, R@5={:.4f}, R@10={:.4f}, mAP={:.4f}'.format(
        metrics['R@1'], metrics['R@5'], metrics['R@10'], metrics['mAP']))
    if metrics['AUROC'] is not None:
        print('OOD: AUROC={:.4f}, FPR@TPR95={:.4f}, Recall@1_Seen={:.4f}, Recall@1_Unseen={:.4f}'.format(
            metrics['AUROC'], metrics['FPR@TPR95'], 
            metrics['Recall@1_Seen'] or 0, metrics['Recall@1_Unseen'] or 0))

    return metrics['mAP'], metrics['R@1']


def main():
    args = get_parser().parse_args()

    if args.dataset == 'ip102':
        if args.dataset_root == 'datasets/market1501':
            args.dataset_root = find_data_root([
                'D:/Sau_Benh_object/retrieval-img/IP102 dataset',
                '/kaggle/input/ip102',
                './IP102 dataset',
                '../IP102 dataset'
            ], env_var='IP102_DATA_ROOT')
        
        if args.filtered_class_path is None:
            args.filtered_class_path = find_file_deep('filtered_class.txt', [
                args.dataset_root,
                'D:/Sau_Benh_object/retrieval-img/IP102 dataset',
                '/kaggle/input/ip102',
                './IP102 dataset',
                '../IP102 dataset'
            ])
        
        if args.classes_txt_path is None:
            args.classes_txt_path = find_file_deep('classes.txt', [
                args.dataset_root,
                'D:/Sau_Benh_object/retrieval-img/IP102 dataset',
                '/kaggle/input/ip102',
                './IP102 dataset',
                '../IP102 dataset'
            ])
        
        print(f'Dataset root: {args.dataset_root}')
        print(f'Filtered classes: {args.filtered_class_path}')
        print(f'Classes txt: {args.classes_txt_path}')

    if not os.path.exists(args.exp_root):
        os.makedirs(args.exp_root)

    if torch.cuda.is_available() and not args.cuda:
        print("\nStrongly recommend to run with '--cuda' if you have a device with CUDA support.")

    print('='*40)
    print('Dataset: {}'.format(args.dataset))
    print('Model: ResNet-50')
    print('Optimizer: Adam')
    print('Image height: {}'.format(args.img_height))
    print('Image width: {}'.format(args.img_width))
    print('Loss: {}'.format(args.loss_type))
    if args.loss_type in ['dmml']:
        print('  margin: {}'.format(args.margin))
    print('  class number: {}'.format(args.num_classes))
    if args.loss_type == 'dmml':
        print('  support number: {}'.format(args.num_support))
        print('  query number: {}'.format(args.num_query))
        print('  distance_mode: {}'.format(args.distance_mode))
    else:
        print('  instance number: {}'.format(args.num_instances))
    print('Epochs: {}'.format(args.num_epochs))
    print('Learning rate: {}'.format(args.lr))
    print('  decay beginning epoch: {}'.format(args.lr_decay_start_epoch))
    print('Weight decay: {}'.format(args.weight_decay))
    if args.cuda:
        print('GPU(s): {}'.format(args.gpu))
    print('='*40)

    if args.cuda:
        gids = get_gpu_ids(args.gpu)
        if not gids:
            gids = auto_detect_gpus()
    else:
        gids = None

    if args.manual_seed is None:
        args.manual_seed = int(time.time())
    args.manual_seed = int(args.manual_seed)
    init_seed(args, gids)
    print(f'seed is set to {args.manual_seed}.')

    if args.dataset == 'market1501':
        TOTAL_TASK_NUM = 751
        BASE_CLS_NUM = 76
        TASK_NUM = 10
        task_sizes = [BASE_CLS_NUM] + [75] * (TASK_NUM - 1)
    elif args.dataset == 'ip102':
        TASK_NUM = 4
        task_sizes = [7, 6, 6, 6]
        TOTAL_TASK_NUM = sum(task_sizes)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported')
    
    args.task_classes = []
    start = 0
    for size in task_sizes:
        args.task_classes.append(list(range(start, start + size)))
        start += size
    
    print(f'Task splits: {args.task_classes}')

    logger = MetricsLogger(os.path.join(args.exp_root, args.method))
    all_task_maps = []

    for task_id in range(args.start_task_id, TASK_NUM):
        model = make_model(args, gids)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = make_loss(args, gids)
        
        if task_id > 0:
            model.to('cpu')
            model_save_path = os.path.join(args.exp_root, args.method,
                                           '{}_model_last_task_{}.pth'.format(args.method, task_id - 1))
            print('load ckpt from {}'.format(model_save_path))
            model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

            if gids is not None:
                model = model.cuda(gids[0])
                if len(gids) > 1:
                    model = DataParallel(model, gids)

            old_model = copy.deepcopy(model)
            for _, para in old_model.named_parameters():
                para.requires_grad = False
            old_model.eval()
            print('load ckpt done!')

            if not os.path.exists(os.path.join(args.exp_root, args.method)):
                os.makedirs(os.path.join(args.exp_root, args.method))

            print(f'Starting training task {task_id} ...')
            mAP, Rank_1 = train(args, model, optimizer, criterion, task_id, gids, 
                               old_model=old_model, logger=logger, all_task_maps=all_task_maps)
        else:
            if gids is not None:
                model = model.cuda(gids[0])
                if len(gids) > 1:
                    model = DataParallel(model, gids)
            mAP, Rank_1 = train(args, model, optimizer, criterion, task_id, gids,
                               logger=logger, all_task_maps=all_task_maps)

        print('After learning TASK {}, the mAP is {:.4f} and R@1 is {:.4f}'.format(task_id, mAP, Rank_1))
        print(f'Training {task_id} completed.')


if __name__ == '__main__':
    main()