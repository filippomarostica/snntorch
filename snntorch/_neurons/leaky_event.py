from .neurons import LIF
import torch
from torch import nn
import numpy as np

# Framework used to manage the ceration of shift ammount
import math
import time

class LeakyEvent(LIF):

    def __init__(
        self,
        beta,
        exp_mode="beta",
        refractory=False,
        refractory_time=5e5,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

        # Custom processing
        self.counter            = torch.zeros(1)
        self.exp_mode           = exp_mode
        self.refractory_time    = refractory_time
        self.refractory         = refractory
        self.last_spike_time    = 0

        # Number of elements in the list
        num_elements = 60
        self.refractory_flag = False
        # LUT containing the beta^n representing the decay at time 'n'
        self.LUT = torch.tensor([self.beta.clamp(0,1)**x for x in range(num_elements)], dtype=torch.float32)
        # shift ammount and balance coefficients are a strategy for emulating HW process on  SW simulation
        self.shift_amounts, self.balance_coefficients = self._find_closest_shift_and_balance(self.LUT)

    def _init_mem(self):
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def init_leaky(self):
        """Deprecated, use :class:`Leaky.reset_mem` instead"""
        return self.reset_mem()

    def forward(self, input_, mem=None):

        if torch.any(input_ > 0):
            # Refractory management
            if self.refractory_flag and self.refractory:
                if time.time_ns() - self.last_spike_time < self.refractory_time:
                    self.counter += 1
                    spk = torch.zeros_like(input_)
                    self.refractory_flag = True
                    # End code execution
                    if self.output:
                        return spk, self.mem
                    elif self.init_hidden:
                        return spk
                    else:
                        return spk, self.mem
                else:
                    self.refractory_flag = False

            if not mem == None:
                self.mem = mem

            if self.init_hidden and not mem == None:
                raise TypeError(
                    "`mem` should not be passed as an argument while `init_hidden=True`"
                )

            if not self.mem.shape == input_.shape:
                self.mem = torch.zeros_like(input_, device=self.mem.device)

            self.reset = self.mem_reset(self.mem)
            self.mem = self.state_function(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                spk = self.fire_inhibition(
                    self.mem.size(0), self.mem
                )  # batch_size
            else:
                spk = self.fire(self.mem)

            if not self.reset_delay:
                do_reset = (
                    spk / self.graded_spikes_factor - self.reset
                )  # avoid double reset
                if self.reset_mechanism_val == 0:  # reset by subtraction
                    self.mem = self.mem - do_reset * self.threshold
                elif self.reset_mechanism_val == 1:  # reset to zero
                    self.mem = self.mem - do_reset * self.mem

            # If the neuron emits a spike refractory_flag must be set to start refractory time
            if torch.any(spk) and self.refractory:
                self.refractory_flag = True
                self.last_spike_time = time.time_ns()

        else:
            self.counter += 1
            spk = torch.zeros_like(input_)

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem

    def _base_state_function(self, input_):
        index = int(self.counter.item())
        # Check exponential decay method
        match self.exp_mode:
            case "beta":
                #base_fn = self.beta.clamp(0, 1) * self.mem + input_
                if 0 <= index < len(self.LUT):
                    base_fn = self.mem * self.LUT[index] + input_
                else:
                    base_fn = input_
            case "shift":
                if 0 <= index < len(self.LUT):
                    base_fn = self.mem * (2**self.shift_amounts[index]) + input_
                else:
                    base_fn = input_
            case "shift_add":
                if 0 <= index < len(self.LUT):
                    base_fn = (self.mem * ((2**self.shift_amounts[index]) + self.balance_coefficients[index])) + input_
                else:
                    base_fn = input_
            case _:
                return "Error exponential mode is incorrect"

        self.counter = torch.zeros(1)
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    @staticmethod
    def _find_closest_shift_and_balance(arr):
        shift_amounts = []
        balance_coefficients = []

        for num in arr:
            if num <= 0:
                raise ValueError("Numbers must be positive.")

            # Find the closest powers of two
            power_down = 2**math.floor(math.log2(num))  # Closest lower power of 2
            power_up = 2**math.ceil(math.log2(num))    # Closest higher power of 2

            # Choose the closest power of two
            if abs(num - power_down) <= abs(num - power_up):
                closest_power = power_down
            else:
                closest_power = power_up

            # Calculate the shift amount and balance coefficient
            shift = int(math.log2(closest_power))
            balance = num - closest_power

            # Store results
            shift_amounts.append(shift)
            balance_coefficients.append(balance)

        return shift_amounts, balance_coefficients

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], LeakyEvent):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], LeakyEvent):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
