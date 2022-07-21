import torch


class AccumuDict(dict):
    """Enhanced dict class for accumulation of PyTorch Tensors."""
    def __init__(self):
        super().__init__()
        self._num = 0
    
    @torch.no_grad()
    def accumulate(self, d):
        if len(self) == 0:
            self.update(d)
        else:
            assert self.keys() == d.keys()

            for k in self.keys():
                self[k] = self[k] + d[k]

        self._num += 1

    @torch.no_grad()
    def mean(self):
        mean_dict = dict()
        for k in self.keys():
            mean_dict[k] = self[k] / self._num
        return mean_dict
    
    def apply(self, func):
        for k in self.keys():
            self[k] = func(self[k])


if __name__ == '__main__':
    d = AccumuDict()
    