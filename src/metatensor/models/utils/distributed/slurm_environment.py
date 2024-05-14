import os
import hostlist


class DistributedEnvironment():
    def __init__(self):
        self._setup_distr_env()
        self.master_addr = os.environ['MASTER_ADDR']
        self.master_port = os.environ['MASTER_PORT']
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])

    def _setup_distr_env(self):
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = '39591'
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
