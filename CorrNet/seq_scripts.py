import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from mmpose.core.evaluation.top_down_eval import pose_pck_accuracy, keypoint_pck_accuracy

def seq_train(loader, model, optimizer, device, epoch_idx, recoder, work_dir, alpha, beta, conf_loss_mode):
    #EDIT Fix loss value
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    accum_steps = 16   #2 for development. 16 for training
    max_grad_norm = 1.0
    save_every = 10000

    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        pose_weights = device.data_to_device(data[4])
        if len(data) > 5: 
            text_tokens = device.data_to_device(data[5])
            mask = device.data_to_device(data[6])
            token_length = device.data_to_device(data[7])
            vn_idxs = device.data_to_device(data[8])
            vn_len = device.data_to_device(data[9])

        #optimizer.zero_grad()
        with autocast():
            breakpoint()
            if len(data) < 6: 
                ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            else:
                #breakpoint()
                ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt, tokens = text_tokens, mask = mask, token_length = token_length, vn_idxs = vn_idxs, vn_len = vn_len)
            predict_heatmap = ret_dict['predict_heatmap']
            #loss = model.keypoint_loss(predict_heatmap, label)
            if not isinstance(label, torch.Tensor):
                label = torch.stack(label,dim=0)
            #breakpoint()
            if len(data) < 6:
                loss = model.keypoint_loss(predict_heatmap, label, pose_weights, alpha, beta)#, conf_loss_mode)
            else:
                #breakpoint()
                glofe_loss = ret_dict['glofe_outputs']['loss']
                predict_heatmap = ret_dict['predict_heatmap']
                breakpoint()
                keypoint_loss = model.keypoint_loss(predict_heatmap, label, pose_weights, alpha, beta)
                #turn into a weighted loss
                loss = keypoint_loss + glofe_loss
            print("training loss:", loss)
            if type(loss) is dict:
                loss = loss['reg_loss'] / accum_steps
            else:
                loss = loss / accum_steps

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            #print(str(data[1])+'  frames')
            #print(str(data[3])+'  glosses')
            optimizer.zero_grad(set_to_none=True)
            scaler = torch.cuda.amp.GradScaler()
            del vid, label, label_lgt, pose_weights, ret_dict, loss
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            #breakpoint()
            scaler.unscale_(optimizer.optimizer)    
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer.optimizer)
            scaler.update()
            optimizer.zero_grad()
            print("reset gradients")
        #nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        loss_value.append(loss.item())

        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))

        save_dir = work_dir
        if (batch_idx + 1) % save_every == 0:
            ckpt_path = os.path.join(
                save_dir,
                f"epoch{epoch_idx}_step{batch_idx+1}.pt"
            )
            torch.save({
                "epoch": epoch_idx,
                "step": batch_idx+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")    
        
        del ret_dict
        del loss

        #Handle leftover grads
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer.optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer.optimizer)
        scaler.update()
        optimizer.zero_grad()

    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return 


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    avg_acc_accum = 0
    evaluations = 0

    total_loss = 0
    num_batches = 0
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        #print("padded pose shape: ", label.shape)
        label_lgt = device.data_to_device(data[3])
        pose_weights = device.data_to_device(data[4])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            predict_heatmap = ret_dict['predict_heatmap']
            #loss = model.keypoint_loss(predict_heatmap, label)
            #breakpoint()
            label = torch.stack(label,dim=0)
            #Get validation loss
            loss = model.keypoint_loss(predict_heatmap, label, pose_weights)
            total_loss += loss['reg_loss'].item()
            num_batches += 1
            print("validation loss:", loss)

        #Extract predicted pose heatmaps from ret_dict
        #predicted_heatmaps = ret_dict['predict_heatmap']
        #save_aggregated_heatmap(predicted_heatmaps[0])
        #predicted_heatmaps_np = predicted_heatmaps.cpu().numpy()

        #Prepare ground truth heatmap data
        #label_np = torch.stack(label, dim=0).cpu().numpy()
        '''
        label_np = label.cpu().numpy()
        predicted_heatmaps = predicted_heatmaps_np[0]
        label_np = label_np[0]
        N = predicted_heatmaps.shape[0]
        normalize = np.tile(np.array([[224, 224]]), (N,1))
        
        mask = label_np[..., 2] > 0.5
        gt = label_np[..., :2]
        if predicted_heatmaps.shape[-1] >= 2:
        # take only x and y
            predicted_heatmaps = predicted_heatmaps[:, :, :2]
        
        acc, avg_acc, cnt = keypoint_pck_accuracy(
            pred = predicted_heatmaps,
            gt = gt,
            mask = mask,
            thr=0.05,
            normalize = normalize
        )
        
        evaluations = evaluations + 1
        avg_acc_accum = avg_acc_accum + avg_acc
        print(avg_acc)
        '''
    mean_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print("Mean validation loss:", mean_val_loss)
    #eval_score = avg_acc_accum / evaluations
    eval_score = mean_val_loss
    return eval_score
    #breakpoint()
    #return avg_acc

 #Function for visualizing heatmaps   
def save_aggregated_heatmap(label, output_path='/data/group1/z40575r/CorrNet_pose_distillation/keypoint_heatmaps_aggregated_eval'):
    label = torch.tensor(label)  # Ensure it's a tensor

    aggregated = label.sum(dim=0)  # Shape: (72, 96)
    aggregated = aggregated / aggregated.max()  # Normalize to [0, 1]

    plt.imshow(aggregated.cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    '''
        total_info += [file_name.split("|")[0] for file_name in data[-1]] #originally broke when .split was called
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']
    try:
        python_eval = True if evaluate_tool == "python" else False
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
                   total_conv_sent)
        conv_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
            triplet=True,
        )
    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
    del conv_ret
    del total_sent
    del total_info
    del total_conv_sent
    del vid
    del vid_lgt
    del label
    del label_lgt
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return lstm_ret
    '''

def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
