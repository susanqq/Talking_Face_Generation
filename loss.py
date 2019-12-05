import torch
import torch.nn as nn
import sys


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input, target, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        # logsoftmax = nn.LogSoftmax()

        if size_average:
            return torch.mean(torch.sum(-target * input.log(), dim=1))
        else:
            return torch.sum(torch.sum(-target * input.log(), dim=1))


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.criterionGAN = nn.BCEWithLogitsLoss()

    def forward(self, inputs, is_real):
        gpu_id = inputs.get_device()
        if is_real:
            target = torch.FloatTensor(inputs.size()).cuda(gpu_id).fill_(1.0)
        else:
            target = torch.FloatTensor(inputs.size()).cuda(gpu_id).fill_(0.0)
        return self.criterionGAN(inputs, target)


class GAN_LR_Loss(nn.Module):
    def __init__(self):
        print("************ GAN LR LOSS **********")
        super(GAN_LR_Loss, self).__init__()
        self.criterionL2 = nn.MSELoss(reduce=False)
        self.criterionCE = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, is_real, targets=None):
        gpu_id = inputs.get_device()
        if is_real:
            return self.criterionCE(inputs, targets)
        else:
            mask = torch.ge(targets, 0) # (batch_size,)
            mask = mask.unsqueeze(-1).repeat(1, inputs.shape[1])
            targets = torch.FloatTensor(inputs.size()).cuda(gpu_id).fill_(0.002)
            loss = self.criterionL2(inputs, targets)

            loss = torch.sum(loss*mask.float())/(inputs.shape[0]*inputs.shape[1])

            return loss



class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, inputs, gt, valid_len=None):
        batch_size = inputs.shape[0]
        if len(inputs.shape)==5: # batch_size, seq_len, c, h, w
            total_len = 0
            for i in range(batch_size):
                total_len += valid_len[i].item()
            valid_inputs = inputs.new_zeros(total_len, inputs.shape[2], inputs.shape[3], inputs.shape[4])
            valid_gt = gt.new_zeros(total_len, gt.shape[2], gt.shape[3], gt.shape[4])
            idx = 0
            for i in range(batch_size):
                valid_inputs[idx:idx+valid_len[i]] = inputs[i,0:valid_len[i]]
                valid_gt[idx:idx+valid_len[i]] = gt[i,0:valid_len[i]]
                idx = idx+valid_len[i]
            return self.criterion(valid_inputs, valid_gt)
        else:
            return self.criterion(inputs, gt)




class LipReadLoss(nn.Module):
    def __init__(self, criterion='l1'):
        super(LipReadLoss, self).__init__()
        self.criterion = criterion
        if criterion == 'l1':
            self.criterionL1 = nn.L1Loss()
        elif criterion == 'l2':
            self.criterionL2 = nn.MSELoss()
        elif criterion == 'KL':
            self.criterionKL = nn.KLDivLoss()
        elif criterion == 'CE':
            self.criterionCE = nn.CrossEntropyLoss()
        elif criterion == 'soft_CE':
            self.criterionCE_soft = SoftCrossEntropy()


    def forward(self, fake, real):

        # score, pred = real.topk(5, 1, True, True)
        # fake_score =  torch.gather(fake, 1, pred)
        # print(score, fake_score)

        if self.criterion=='l1':
            return self.criterionL1(fake, real)
        elif self.criterion=='l2':
            return self.criterionL2(fake, real)
        elif self.criterion=='KL':
            return self.criterionKL(fake.log(), real)
        elif self.criterion=='CE':
            return self.criterionCE(fake, real)
        elif self.criterion=='soft_CE':
            return self.criterionCE_soft(fake, real)

