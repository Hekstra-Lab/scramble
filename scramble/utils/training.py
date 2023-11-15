

class OnlineMean():
    """
    A small class to compute the mean of metrics online
    during training.
    """
    def __init__(self, fmt_str='0.2e'):
        self.count = 0
        self.mean = None
        self.fmt_str = fmt_str

    def format(self, number):
        return ('{val:' + self.fmt_str + '}').format(val=number)

    def __call__(self, metrics : dict) -> dict:
        """
        Parameters
        ----------
        metrics : dict
            a dictionary of {metric_name (string) : metric_value (float)} entries

        Returns
        -------
        means : diction
            a dictionary of {metric_name (string) : running_mean (float)} entries
        """
        self.count += 1
        if self.mean is None:
            self.mean = metrics
        else:
            for k,v in self.mean.items():
                self.mean[k] += (metrics[k] - v) / self.count

        out = {k:self.format(v) for k,v in self.mean.items()}
        return out


