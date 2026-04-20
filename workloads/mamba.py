import torch

from workloads.base import FixtureBase, WorkloadBase


class DaCumsumFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, num_chunks, chunk_len, n_heads, tune", [
                pytest.param(1, 2, 64, 4, False, marks=pytest.mark.smoke),
                pytest.param(2, 4, 64, 8, False, marks=pytest.mark.full),
                pytest.param(1, 2, 128, 4, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 16, False, marks=pytest.mark.full),
            ]),
        ]

class DaCumsumFwdTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads

    def gen_inputs(self):
        b, C, Q, h = self.batch, self.num_chunks, self.chunk_len, self.n_heads
        seq_len = C * Q
        # dt > 0 (softplus output in Mamba-2), A <= 0 (negative decay)
        dt = torch.rand(b, seq_len, h, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        A = -torch.rand(h, dtype=torch.float32, device="cuda")
        return dt, A


class SSDChunkScanFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune", [
                pytest.param(1, 2,  64, 4, 64,  32, 1, torch.float16,  False, marks=pytest.mark.smoke),
                pytest.param(2, 4,  64, 8, 64,  64, 2, torch.float16,  False, marks=pytest.mark.full),
                pytest.param(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 2,  64, 4, 64,  32, 2, torch.bfloat16, False, marks=pytest.mark.full),
            ]),
        ]


class SSDChunkScanFwdTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, c, L, h, p, n, g = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        S = c * L

        # Official layouts (aligned with _chunk_scan_fwd in mamba_ssm)
        x           = torch.randn(b, S, h, p,    dtype=self.dtype,    device="cuda") * 0.1
        cb          = torch.randn(b, c, g, L, L, dtype=self.dtype,    device="cuda") * 0.1
        dA_cumsum   = -torch.rand(b, h, c, L,    dtype=torch.float32, device="cuda").cumsum(-1)
        C           = torch.randn(b, S, g, n,    dtype=self.dtype,    device="cuda") * 0.1
        prev_states = torch.randn(b, c, h, p, n, dtype=self.dtype,    device="cuda") * 0.1
        dt          = torch.rand( b, h, c, L,    dtype=self.dtype,    device="cuda") * 0.1 + 0.01
        return x, cb, dA_cumsum, C, prev_states, dt

class SSDChunkStateFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx", [
                pytest.param(
                    1, 2, 64, 4, 64, 32, 1, torch.float16, False, False, marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2, 4, 64, 8, 64, 64, 2, torch.float16, False, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 2, 128, 4, 128, 32, 1, torch.bfloat16, False, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 2, 64, 4, 64, 32, 2, torch.bfloat16, False, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 4, 64, 8, 64, 64, 2, torch.float16, False, True, marks=pytest.mark.full,
                ),
            ]),
        ]

class SSDChunkStateFwdTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        has_seq_idx: bool = False,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.has_seq_idx = has_seq_idx

    def gen_inputs(self):
        b, c, Q, h, p, n, g = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        seq_len = c * Q
        x = torch.randn(b, seq_len, h, p, dtype=self.dtype, device="cuda") * 0.1
        Bmat = torch.randn(b, seq_len, g, n, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: monotonically non-increasing (negative values, cumsum of negatives)
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
        dt = torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        seq_idx = None
        if self.has_seq_idx:
            # simulate two packed sequences per batch row, split at midpoint
            seq_idx = torch.zeros(b, seq_len, dtype=torch.int32, device="cuda")
            seq_idx[:, seq_len // 2:] = 1
        return x, Bmat, dt, dA_cumsum, seq_idx

class SSDDecodeFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, n_heads, d_head, d_state, n_groups, dtype, tune", [
                pytest.param(
                    1, 4, 64, 16, 1, torch.float16, False, marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2, 8, 64, 32, 2, torch.float16, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 4, 64, 16, 1, torch.bfloat16, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 8, 128, 64, 4, torch.bfloat16, False, marks=pytest.mark.full,
                ),
            ]),
        ]

class SSDDecodeTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, h, p, n, g = (
            self.batch, self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        # A <= 0 (negative decay), dt > 0 (post-softplus)
        A = -torch.rand(h, p, n, dtype=torch.float32, device="cuda")
        dt = torch.rand(b, h, p, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        x = torch.randn(b, h, p, dtype=self.dtype, device="cuda") * 0.1
        B_in = torch.randn(b, g, n, dtype=self.dtype, device="cuda") * 0.1
        C_in = torch.randn(b, g, n, dtype=self.dtype, device="cuda") * 0.1
        state = torch.randn(b, h, p, n, dtype=torch.float32, device="cuda") * 0.1
        return A, dt, x, B_in, C_in, state

class SSDStatePassingFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, num_chunks, n_heads, d_state, dtype, tune", [
                pytest.param(1, 2,  4,  32, torch.float16,  False, marks=pytest.mark.smoke),
                pytest.param(2, 4,  8,  64, torch.float16,  False, marks=pytest.mark.full),
                pytest.param(1, 2,  4,  32, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 4,  8,  64, torch.bfloat16, False, marks=pytest.mark.full),
            ]),
        ]

class SSDStatePassingFwdTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_state: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_state = d_state
        self.dtype = dtype

    def gen_inputs(self):
        b, c, h, d = self.batch, self.num_chunks, self.n_heads, self.d_state
        states = torch.randn(b, c, h, d, dtype=self.dtype, device="cuda") * 0.1
        dA_chunk_cumsum = -torch.rand(b, h, c, dtype=torch.float32, device="cuda").cumsum(-1)
        initial_states = torch.randn(b, h, d, dtype=torch.float32, device="cuda") * 0.1
        return states, dA_chunk_cumsum, initial_states
