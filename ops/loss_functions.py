import torch
import torch.nn.functional as F


# output, features, centers, dist, dist_logits, target_var

def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True))/features.shape[0]

    return distance


def dce_loss(output, features, centers, dist_logits, labels, reg):

    loss1 = F.nll_loss(dist_logits, labels)
    loss2 = regularization(features, centers, labels)

    loss = loss1 + reg*loss2
    return loss


def nll_dce_loss(output, features, centers, dist_logits, labels, reg):

    criterion = torch.nn.CrossEntropyLoss().cuda()
    loss1 = dce_loss(output, features, centers, dist_logits, labels, reg)
    loss2 = criterion(output, labels)

    loss = 0.01*loss1 + loss2
    return loss


# margin based classification loss (MCL)
def mcl_loss(output, features, centers, dist_logits, labels, reg):
    pass


# generalized margin based classification loss (GMCL)
def gmcl_loss(output, features, centers, dist_logits, labels, reg):
    pass


# minimum classification error loss (MCE)
def mce_loss(output, features, centers, dist_logits, labels, reg):
    pass


def get_criterion(loss_type):
    loss_functions = {
        'cpl_dce_loss': dce_loss,
        'cpl_nll_dce_loss': nll_dce_loss,
    }
    return loss_functions[loss_type]