"""Tests for elementwise kernel strategy validation.

Verifies that UnaryKernel and BinaryKernel raise ValueError when
instantiated with an invalid strategy string.
"""

import pytest
import torch

from tileops.kernels.elementwise import AddKernel, ReluKernel


@pytest.mark.smoke
class TestUnaryKernelStrategyValidation:
    """UnaryKernel.__init__ must raise ValueError for invalid strategies."""

    def test_invalid_strategy_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown strategy 'invalid'"):
            ReluKernel(N_total=1024, dtype=torch.float16, strategy="invalid")

    def test_typo_strategy_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown strategy 'registar_copy'"):
            ReluKernel(N_total=1024, dtype=torch.float16, strategy="registar_copy")

    def test_error_message_lists_valid_strategies(self):
        with pytest.raises(ValueError, match="expected one of"):
            ReluKernel(N_total=1024, dtype=torch.float16, strategy="nonexistent")

    @pytest.mark.parametrize("strategy", ReluKernel.STRATEGIES)
    def test_valid_strategies_do_not_raise(self, strategy):
        """Valid strategies must not raise any exception during construction."""
        ReluKernel(N_total=1024, dtype=torch.float16, strategy=strategy)


@pytest.mark.smoke
class TestBinaryKernelStrategyValidation:
    """BinaryKernel.__init__ must raise ValueError for invalid strategies."""

    def test_invalid_strategy_raises_valueerror(self):
        shape = (1024,)
        with pytest.raises(ValueError, match="Unknown strategy 'invalid'"):
            AddKernel(
                N_total=1024, dtype=torch.float16,
                coalesced_shape=shape, a_strides=(1,), b_strides=(1,),
                a_numel=1024, b_numel=1024, strategy="invalid",
            )

    def test_typo_strategy_raises_valueerror(self):
        shape = (1024,)
        with pytest.raises(ValueError, match="Unknown strategy 'explicitt_parallel'"):
            AddKernel(
                N_total=1024, dtype=torch.float16,
                coalesced_shape=shape, a_strides=(1,), b_strides=(1,),
                a_numel=1024, b_numel=1024, strategy="explicitt_parallel",
            )

    def test_error_message_lists_valid_strategies(self):
        shape = (1024,)
        with pytest.raises(ValueError, match="expected one of"):
            AddKernel(
                N_total=1024, dtype=torch.float16,
                coalesced_shape=shape, a_strides=(1,), b_strides=(1,),
                a_numel=1024, b_numel=1024, strategy="nonexistent",
            )

    def test_register_copy_rejected_for_binary(self):
        """BinaryKernel does not support register_copy strategy."""
        shape = (1024,)
        with pytest.raises(ValueError, match="Unknown strategy 'register_copy'"):
            AddKernel(
                N_total=1024, dtype=torch.float16,
                coalesced_shape=shape, a_strides=(1,), b_strides=(1,),
                a_numel=1024, b_numel=1024, strategy="register_copy",
            )

    @pytest.mark.parametrize("strategy", AddKernel.STRATEGIES)
    def test_valid_strategies_do_not_raise(self, strategy):
        """Valid strategies must not raise any exception during construction."""
        shape = (1024,)
        AddKernel(
            N_total=1024, dtype=torch.float16,
            coalesced_shape=shape, a_strides=(1,), b_strides=(1,),
            a_numel=1024, b_numel=1024, strategy=strategy,
        )
