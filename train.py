import os
import numpy as np
import yaml
import time
import torch
# import random
import warnings
from torch.optim import lr_scheduler
import models.utils as utils
from tqdm import tqdm
from models.line_transformer import LineTransformer
from dataloaders.dataloader import LineDescDataset
from evaluations import criteria
from evaluations.metric import Result, AverageMeter

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

warnings.filterwarnings("ignore")

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_everything(seed=1004):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Dict2Class(object):
    def __init__(self, dic):
        for key, val in dic.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Dict2Class(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Dict2Class(val) if isinstance(val, dict) else val)

def main():
    time_stst = time.time()
    
    with open('./train_manager.yaml', 'r') as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)
    conf = Dict2Class(conf_dict)

    ##
    logger = utils.logger(conf, tensorboard_writer=writer)
    device = torch.device('cuda:'+str(conf.device[0]) if torch.cuda.is_available() else "cpu")
    if conf.fix_randomness:
        seed_everything()
    if conf.ignore_warnings:
        warnings.filterwarnings("ignore")

    datapath = conf_dict[conf.dataset_type]['data_path']
    output_directory = conf_dict['backup_path']
    if conf.mode == 'train':
        train_path = [os.path.join(path, 'train') for path in datapath]
        val_path = [os.path.join(path, 'val') for path in datapath]
        train_set = LineDescDataset(train_path)
        val_set = LineDescDataset(val_path)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=conf.batch_size, num_workers=conf.num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, batch_size=conf.batch_size, num_workers=conf.num_workers, pin_memory=False)

        model = LineTransformer(conf_dict['linetr']).to(device)
        
        if conf_dict[conf.dataset_type]['resume']:
            checkpoint = torch.load(conf_dict[conf.dataset_type]['checkpoint_path'], map_location=device)

            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            filtered_update_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(filtered_update_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            
        model = torch.nn.DataParallel(model, device_ids=conf.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(conf.lr), weight_decay=conf.wd)
        scheduler = None
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999, last_epoch=-1)

        try:
            for epoch in range(0, conf.epochs):
                is_best = False
                print("starting training epoch {} ..".format(epoch))
                run_epochs("train", conf, train_loader, model, optimizer, logger, epoch, device, scheduler)

                # # evaluate on validation set
                is_best = run_epochs("val", conf, val_loader, model, None, logger, epoch, device, lr_scheduler)  
                
                ## save model      
                utils.save_checkpoint({ # save checkpoint
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best, epoch, logger.output_directory)
                        
        except KeyboardInterrupt:
            print ("press ctrl + c, save model!")
            utils.save_checkpoint({ # save checkpoint
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch, logger.output_directory)
    
    time_end = time.time() - time_stst
    print("Time Consumption:", time_end)

        
train_iter = 0
def run_epochs(mode, args, dataloader, model, optimizer, logger, epoch, device, scheduler):
    global train_iter
    
    block_average_meter = AverageMeter(args)
    average_meter = AverageMeter(args)
    meters = [block_average_meter, average_meter]
    descriptor_loss = criteria.descriptor_loss()
    
    # switch to appropriate mode
    assert mode in ["train", "val", 'test'], "unsupported mode: {}".format(mode)
    
    if mode == 'train':
        model.train()
        args.mode = mode    # conditional_save_info
        lr = -1
    elif mode == 'val':
        args.mode = mode    # conditional_save_info
        with torch.no_grad():
            model.eval()
            lr = -1

    # start working within mini-batch
    time_model = 0
    time_st = time.time()
    for batch_index, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        train_iter += 1

        if args.debug_times:
            print('[0] load batches:', time.time() - time_st)
            time_st = time.time()
        
         # [1] prepare data in GPU
        batch_data = {k:v.to(device=device, dtype=torch.float, non_blocking = True) for k,v in batch_data.items()}

        if args.debug_times:
            print('[1] prepare data in GPU:', time.time() - time_st)
            time_st = time.time()

         ## [2] Compute Loss and back-propagation
        if mode == 'train':
            with torch.set_grad_enabled(True):
                 # matching_threshold = 0.2
                 # batch_data['mat_assign_sublines'][batch_data['mat_assign_sublines'] > matching_threshold] = 1

                data0 = {k[:-1]:v for k,v in batch_data.items() if k[-1]=='0'}
                data1 = {k[:-1]:v for k,v in batch_data.items() if k[-1]=='1'}
                
                time_model_st = time.time()
                pred0 = model(data0)
                pred1 = model(data1)
                
                pred0 = {k+'0':v for k,v in pred0.items()}
                pred1 = {k+'1':v for k,v in pred1.items()}
                pred = {**pred0, **pred1}
                # with writer:
                #     model_wrapper = ModelWrapper(model)
                #     writer.add_graph(model_wrapper,batch_data)  
                # writer.add_graph(torch.jit.trace(model, batch_data, strict=False), batch_data)
                # writer.add_graph(model, batch_data)
                time_model += time.time() - time_model_st

                num_batches, num_lines = batch_data['sublines0'].shape[:2]
                gt_matches_sublines = torch.zeros((num_batches, num_lines+1, num_lines+1), device=device)
                lmatches = batch_data['lmatches'].type(torch.long) #.cpu().numpy().astype(int)
                
                for i, batch in enumerate(lmatches):
                    batch=batch[batch[:,0]!=-1]
                    gt_matches_sublines[i,batch[:,0], batch[:,1]] = 1
                batch_data['mat_assign_sublines'] = gt_matches_sublines #torch.from_numpy(gt_matches_sublines)
                ## loss function ##
                # loss = 0.8 * matching_loss(pred['score_matrix'], batch_data['assign_mat_gt'])
                # loss = matching_loss(pred['score_matrix_line'], batch_data['mat_assign_sublines'])
                # loss = loss + 0.2 * matching_loss(pred['score_matrix_line'], batch_data['mat_assign_sublines'])
                loss, hardest_positive, hardest_negative= descriptor_loss(pred, batch_data)

                print(hardest_positive.item(), hardest_negative.item(), loss.item())

                ## backprop ##
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lr = utils.adjust_learning_rate(args.lr, scheduler, train_iter)

        elif mode == 'val': 
            with torch.no_grad():
                time_model_st = time.time()
                
                data0 = {k[:-1]:v for k,v in batch_data.items() if k[-1]=='0'}
                data1 = {k[:-1]:v for k,v in batch_data.items() if k[-1]=='1'}
                
                time_model_st = time.time()
                pred0 = model(data0)
                pred1 = model(data1)
                
                pred0 = {k+'0':v for k,v in pred0.items()}
                pred1 = {k+'1':v for k,v in pred1.items()}
                pred = {**pred0, **pred1}
                
                num_batches, num_lines = batch_data['sublines0'].shape[:2]
                gt_matches_sublines = torch.zeros((num_batches, num_lines+1, num_lines+1), device=device)
                lmatches = batch_data['lmatches'].type(torch.long) #.cpu().numpy().astype(int)
                
                for i, batch in enumerate(lmatches):
                    batch=batch[batch[:,0]!=-1]
                    gt_matches_sublines[i,batch[:,0], batch[:,1]] = 1
                batch_data['mat_assign_sublines'] = gt_matches_sublines #torch.from_numpy(gt_matches_sublines)
                time_model += time.time() - time_model_st

                loss, hardest_positive, hardest_negative = descriptor_loss(pred, batch_data)



        if args.debug_times:
            print('[2] inference and loss:', time.time() - time_st)
            time_st = time.time()

        ## [3] evaluation and logging
        result = Result(mode, args)
        result.evaluate(pred, batch_data, loss.item(), batch_index)

        writer.add_scalar('positive/'+mode, hardest_positive.item(), batch_index+len(dataloader)*epoch)
        writer.add_scalar('negative/'+mode, hardest_negative.item(), batch_index+len(dataloader)*epoch)
        writer.add_scalar('loss/'+mode, loss.item(), batch_index+len(dataloader)*epoch)
        writer.add_scalar('precision/'+mode, np.mean(result.precision), batch_index+len(dataloader)*epoch)
        writer.add_scalar('recall/'+mode, np.mean(result.recall), batch_index+len(dataloader)*epoch)
        writer.add_scalar('f1/'+mode, np.mean(result.f1_score), batch_index+len(dataloader)*epoch)

        batch_size = batch_data['klines0'].shape[0]
        inference_time = time_model/batch_size
        time_model = 0
        for m in meters:
            m.update(result, 0, batch_size)

        logger.conditional_print(args, mode, batch_index, epoch, lr, len(dataloader),
                                block_average_meter, average_meter, loss.item())
        

        if args.debug_times:
            print('[3] logging loss and metrics:', time.time() - time_st)
            time_st = time.time()

    avg = logger.conditional_save_info(args, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    logger.conditional_summarize(args, mode, avg, is_best)
    
    return is_best

if __name__ == '__main__':
    main()