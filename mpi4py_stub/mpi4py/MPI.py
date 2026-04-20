"""Fake MPI module — single-process behavior.

Implements just enough of the mpi4py.MPI API that guided_diffusion uses:
- COMM_WORLD.rank / Get_rank() = 0
- COMM_WORLD.size / Get_size() = 1
- COMM_WORLD.bcast(x, root=0) = x (identity, single proc)
"""


class _FakeComm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, data, root=0):
        # 单进程下广播就是恒等
        return data

    def barrier(self):
        return None

    def allreduce(self, data, op=None):
        return data

    def gather(self, data, root=0):
        return [data]

    def allgather(self, data):
        return [data]


COMM_WORLD = _FakeComm()

# 占位常量，防止有人 import 这些
SUM = "SUM"
MAX = "MAX"
MIN = "MIN"
