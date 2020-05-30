def contain(name_, list_):
    for i in list_:
        if i in name_:
            return True
    return False


def freeze(net, freeze_list=[], unfreeze_list=[]):
    for name, value in net.named_parameters():
        if contain(name, freeze_list):
            value.requires_grad = False
        if contain(name, unfreeze_list):
            value.requires_grad = True


        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

