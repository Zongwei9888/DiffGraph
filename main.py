import torch

import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from Model import GaussianDiffusion,Denoise,GCNModel,GCN_Diffusion_Model

from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *

import os
import logging
import sys



class Coach:
    def __init__(self, handler):
        self.handler = handler
        print('USER',  self.handler.userNum, 'ITEM', self.handler.itemNum)
        print('NUM OF INTERACTIONS', self.handler.train_dataloader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    def makePrintAllK(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %s, ' % (metric, str(val))
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    
    def run(self):
        self.prepareModel()
        log('Model Prepared')
        log('Model Initialized')

        recallMax = 0
        ndcgMax = 0
        bestEpoch = 0

        wait = 0

        #file save setting
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.data + '/'
        log_file = f'{args.data}_hid_{args.latdim}_layer_{args.gcn_layer}_' + \
                           f'lr_{args.lr}_difflr_{args.difflr}_diff_dim_{args.dims}_reg_{args.reg}_batch_{args.batch}_diffstep_{args.steps}_T_dim_{args.d_emb_size}_noise_scale_{args.noise_scale}_'+\
                            f'new_data'
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')  
        args.save_path = args.data + '/'+log_file       



        for ep in range(0, args.epoch):
            tstFlag = (ep % 1 == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if (reses['Recall'] > recallMax):
                    recallMax = reses['Recall']
                    ndcgMax = reses['NDCG']
                    bestEpoch = ep
                    wait = 0
                    self.saveModel()
                else:
                    wait+=1
                # log(self.makePrint('Test', ep, reses, tstFlag))
                logger.info(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
            print()
            if wait >= args.patience:
                print(f'Early stop at epoch {ep}, best epoch {bestEpoch}')
                break
        print('Best epoch : ', bestEpoch, ' , Recall@20 : ', recallMax, ' , NDCG@20 : ', ndcgMax)

    def run_allk(self):
        self.prepareModel()
        log('Model Prepared')
        log('Model Initialized')

        recallMax = 0
        ndcgMax = 0
        bestEpoch = 0

        wait = 0

         #file save setting
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.data + '/'
        log_file = f'{args.data}_hid_{args.latdim}_layer_{args.gcn_layer}_' + \
                           f'lr_{args.lr}_difflr_{args.difflr}_diff_dim_{args.dims}_reg_{args.reg}_batch_{args.batch}_diffstep_{args.steps}_T_dim_{args.d_emb_size}_noise_scale_{args.noise_scale}_'+\
                            f'new_data'
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')  
        args.save_path = args.data + '/'+log_file    

        for ep in range(0, args.epoch):
            tstFlag = (ep % 1 == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpochAllK()
                if (reses['Recall'][1] > recallMax):
                    recallMax = reses['Recall'][1]
                    ndcgMax = reses['NDCG'][1]
                    bestEpoch = ep
                    wait = 0
                else:
                    wait+=1
                # log(self.makePrintAllK('Test', ep, reses, tstFlag))
                logger.info(self.makePrintAllK('Test', ep, reses, tstFlag))
                self.saveHistory()
            print()
            if wait >= args.patience:
                print(f'Early stop at epoch {ep}, best epoch {bestEpoch}')
                break
        print('Best epoch : ', bestEpoch, ' , Recall@20 : ', recallMax, ' , NDCG@20 : ', ndcgMax)



    def prepareModel(self):
      
        self.model = GCN_Diffusion_Model(self.handler).cuda()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=args.decay_step,
            gamma=args.decay
        )


    def trainEpoch(self):
        trnLoader = self.handler.train_dataloader
        
        trnLoader.dataset.negSampling()
        epLoss, epRecLoss, eDiffLoss, eSSLoss = 0, 0, 0 ,0
        steps = trnLoader.dataset.__len__() // args.batch
        mode = 'all_diff'
        for i, tem in enumerate(trnLoader):
        
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            self.opt.zero_grad()
            loss,bprLoss,regLoss,diff_loss = self.model.cal_loss(ancs,poss,negs)
            epLoss += loss.item()
            epRecLoss += bprLoss.item()
            eDiffLoss += diff_loss.item()
            

            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.3f, diffLoss = %.3f,regLoss = %.3f' % (i, steps, loss, diff_loss , regLoss), save=False, oneline=True)
        # self.scheduler.step()
       
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['recLoss'] = epRecLoss / steps
        ret['diffLoss'] = eDiffLoss / steps
        return ret


    def testGroup(self):
        tstLoader = self.handler.test_dataloader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        self.model.eval()

        with torch.no_grad():
            usrEmbeds, itmEmbeds = self.model.predict()
        user_group = self.handler.grup_list
        for key, u_list in user_group.items():

            i += 1
            usr = torch.from_numpy(u_list)
            usr = usr.long().cuda()
            trnMask = self.handler.trainLabel[u_list].toarray()
            trnMask = torch.from_numpy(trnMask).to(torch.float64)
            # trnMask = np.reshape(trnMask, [-1])
            trnMask = trnMask.cuda()
           
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.test_dataloader.dataset.user_pos_lists, usr)
            group_recall,group_ndcg = recall / len(usr),ndcg / len(usr)
            print('Group '+ str(key)+':'+'recall = %.4f, ndcg = %.4f ' % (group_recall,group_ndcg))
            epRecall += recall
            epNdcg += ndcg
            # log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret
    def testEpoch(self):
        tstLoader = self.handler.test_dataloader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        self.model.eval()

        with torch.no_grad():
            usrEmbeds, itmEmbeds = self.model.predict()
            user_emb = usrEmbeds.cpu().numpy()
            item_emb = itmEmbeds.cpu().numpy()
            emb_save = './'+args.data+'hdl_emb.pkl'
            with open(emb_save,'wb') as f:
                pickle.dump({'user_embed':user_emb,'item_embed':item_emb},f)
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
           
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.test_dataloader.dataset.user_pos_lists, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret
    
    

    def testEpochAllK(self):
        tstLoader = self.handler.test_dataloader
        epRecall = [0,0,0] 
        epNdcg = [0,0,0] 
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        eval_topk = [10,20,40]

        with torch.no_grad():
            usrEmbeds, itmEmbeds = self.model.predict()
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
           
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            for i , topk in enumerate(eval_topk):
                _, topLocs = t.topk(allPreds, topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.test_dataloader.dataset.user_pos_lists, usr)
                epRecall[i] += recall
                epNdcg[i] += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = [ round(i/num,5) for i in epRecall]
        ret['NDCG'] = [ round(i/num,4) for i in epNdcg]
        return ret
    


    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        
    def saveModel(self):
        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        
        ckp = t.load('./Models/' + args.load_model )
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        # with open('../History/' + args.load_model + '.his', 'rb') as fs:
        #     self.metrics = pickle.load(fs)
        log('Model Loaded')
    def test(self):
        self.prepareModel()
        log('Model Prepared')
        log('Model Initialized')

        recallMax = 0
        ndcgMax = 0
        bestEpoch = 0
     
        args.load_model = args.data+'/'+'tmall_hid_64_layer_2_lr_0.001_difflr_0.001_diff_dim_[64]_reg_0.03_batch_2048_diffstep_150_T_dim_8_noise_scale_0.0001_hyper_params_test.mod'
        self.loadModel()
        
        reses = self.testEpoch()
                
        log(self.makePrint('Test', 1, reses, 1))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    log('Start')
    handler = DataHandler()
    handler.load_data()
    log('Load Data')

    coach = Coach(handler)
    # coach.run()
    coach.test()