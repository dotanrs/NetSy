import random
import math

DEFAULT_CONNECTION_STRENGTH = 1
ACTIVATION_RANGE_HIGH = 2
ACTIVATION_RANGE_LOW = -1.5
DEFAULT_NOISE_MEAN = 0.5
DEFAULT_DECAY = 0.1
SPIKE_SIZE = 3
DEFAULT_THRESHOLD_P = 0.05
DEFAULT_SIGMOID_MEAN = 1

def get_from_kwargs_or_default(argname, default, **kwargs):
	return kwargs[argname] if argname in kwargs else default


class Utils:

	def decay(var, coef):
		return var - var * coef


class Neuron:
	"""
	A simple neuron that transmit the input it recieves
	"""

	def __init__(self, network, index, name = "anonymous neuron", **kwargs):
		range_default = [ACTIVATION_RANGE_LOW, ACTIVATION_RANGE_HIGH]

		self.network = network
		self.index = index

		# The minimal and maximal values of activation the neuron can have
		self.range = get_from_kwargs_or_default("range", range_default, **kwargs)

		# If set, the neuron will day after this many iterations
		self.lifespan = get_from_kwargs_or_default("lifespan", False, **kwargs)

		# If set, the neuron will print logs to console
		self.log = get_from_kwargs_or_default("log", False, **kwargs)

		# The length of the refractory period, if set
		self.refractory_time = get_from_kwargs_or_default("refractory_time", 10, **kwargs)

		# Number of iterations passed
		self.ticks = 0

		self.activation = self.range[0]
		self.name = self._name_prefix + name
		self.input = 0
		self.in_refractory_period = False
		self.refractory_period_timer = 0
		self.is_active = True

	@property
	def _name_prefix(self):
		return "[{0}] ".format(self._type_identifier())

	def _type_identifier(self):
		return "N"

	def _log(self, param, value, second_val=None):
		if not self.log:
			return
		if second_val:
			print("%d. [%s] %s = %s, %s" % (self.ticks, self.name, param, str(value), str(second_val)))
		else:
			print("%d. [%s] %s = %s" % (self.ticks, self.name, param, str(value)))

	def _apply_input(self):
		self._log("Input", self.input)
		self.activation += self.input
		self.input = 0

	def _apply_range(self):
		if self.activation > self.range[1]:
			self.activation = self.range[1]

		if self.activation < self.range[0]:
			self.activation = self.range[0]

	def _calc_output(self):
		"""
		called in each run, not including refractory period
		"""
		self._apply_input()
		self._apply_range()

	def _internal_processes(self):
		"""
		Called in each run, including in refractory period
		"""
		self.activation = 0

	def run(self):
		"""
		This function is assumed to be called exactly once per iteration of the 
		network
		"""
		if not self.is_active:
			self.network.set_activation(self.index, 0)
			return

		self.ticks += 1

		if self.lifespan and self.ticks > self.lifespan:
			self.activation = 0
			self.is_active = False
		
		self._internal_processes()

		if self.in_refractory_period:
			if self.refractory_period_timer > self.refractory_time:
				self.in_refractory_period = False
			else:
				self.refractory_period_timer += 1
				self.activation = self.range[0]
				self._log("in refractory period", self.refractory_period_timer)

		self._calc_output()

		self.network.set_activation(self.index, self.get_signal())


	def show_log(self, val=True):
		self.log = val
		self._log("start log, name", self.name)

	def get_name(self):
		return self.name

	def get_activation(self):
		return self.activation

	def set_activation(self, val):
		self._log("Manual activation", val)
		self.activation = val

	def get_signal(self):
		return self.activation

	def add_input(self, signal):
		self.input += signal

	def send_to(self, neuron, connection_strength = DEFAULT_CONNECTION_STRENGTH):
		self.network.set_connections(neuron, self, connection_strength)

	def listen_to(self, neuron, connection_strength = DEFAULT_CONNECTION_STRENGTH):
		""" Note: if connection was previously defined it will be replaced.
		To change the strength you can also use increase_connection_strength
		"""
		self.network.set_connections(self, neuron, connection_strength)

	def set_lifespan(self, new_value):
		self.lifespan = new_value

	def increase_connection_strength(self, neuron, value):
		self.network.increase_connections(neuron, self)


class ThresholdNeuron(Neuron):
	"""
	The threshold neuron will aggregate activation until it reaches a threshold (self.threshold)
	and then fire a signal (SPIKE_SIZE) and reset the activity
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		default_threshold = self.range[1]
		self.threshold 		   = get_from_kwargs_or_default("threshold", 1, **kwargs)
		self.decay_coefficient = get_from_kwargs_or_default("decay_coefficient", DEFAULT_DECAY, **kwargs)
		self.range[1] = SPIKE_SIZE
		self._log("range", self.range)

	def _type_identifier(self):
		return "TH"

	def get_signal(self):
		if self.activation == SPIKE_SIZE:
			return SPIKE_SIZE
		else:
			return 0

	def _apply_input(self):
		if not self.in_refractory_period:
			self.activation += self.input
			self.input = 0

		if self.activation >= self.threshold:
			self.activation = SPIKE_SIZE
			

	def _internal_processes(self):
		if self.activation >= SPIKE_SIZE:
			self.in_refractory_period = True
			self.refractory_period_timer = 1
			self.activation = self.range[0]

		self.activation = Utils.decay(self.activation, self.decay_coefficient)


class WhiteNoiseNeuron(Neuron):
	"""
	A neuron that sends a random valued signal. The signal is sampled uniformly with span 1
	and mean self.mean
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.mean = get_from_kwargs_or_default("mean", DEFAULT_NOISE_MEAN, **kwargs)

	def _type_identifier(self):
		return "WN"

	def get_signal(self):
		self.activation = random.random() * self.mean * 2
		return self.activation

	def _internal_processes(self):
		pass

class BinaryNoiseNeuron(ThresholdNeuron):
	"""
	A neuron that files a spike (self.range[1]) W.P. self.p. 
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.p = get_from_kwargs_or_default("p", DEFAULT_THRESHOLD_P, **kwargs)
		self.refractory_time = 0

	def _type_identifier(self):
		return "BN"

	def _apply_input(self):
		pass

	def get_signal(self):
		signal = random.random()
		if signal < self.p :
			self.activation = self.range[1]
		else:
			self.activation = self.range[0]
			
		return self.activation


class SigmoidNeuron(Neuron):
	""" 
	A neuron who's output is tanh of it's input
	"""

	def __init__(self, **kwargs):
		self.mean = get_from_kwargs_or_default("mean", DEFAULT_SIGMOID_MEAN, **kwargs)
		super().__init__(**kwargs)
		self.range = [-1 * self.mean, 1 * self.mean]
		self.bias = get_from_kwargs_or_default("bias", 0, **kwargs)
		self.tanh_bias = get_from_kwargs_or_default("tanh_bias", 0, **kwargs)
		self.activation = get_from_kwargs_or_default("init", self.range[0], **kwargs)
		self.der_step = get_from_kwargs_or_default("der_step", 0.001, **kwargs)
		# self.activation = self.activation + random.random() * 0.01
		self._log("start", self.activation)
		self.decay_coefficient = 1


	def _type_identifier(self):
		return "SIG"

	def _apply_range(self):
		pass

	def _apply_input(self):
		dif = self.bias + math.tanh(self.input + self.tanh_bias)
		der = -self.activation + dif
		self._log("input, der", self.input, dif)
		self.activation += der * self.der_step
		self.input = 0
		self._log("current", self.activation)

	def get_signal(self):
		return self.activation

	def _internal_processes(self):
		pass # self.activation = Utils.decay(self.activation, self.decay_coefficient)

class LimitSigmoidNeuron(SigmoidNeuron):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.tanh_beta = get_from_kwargs_or_default("tanh_beta", float("inf"), **kwargs)

	def _type_identifier(self):
		return "LSIG"

	def _apply_input(self):

		if self.input != 0:
			inpt_limit = math.tanh(self.tanh_beta * (self.input + self.tanh_bias))
			der = -self.activation + inpt_limit
			self.activation += der * self.der_step
			self.input = 0
		self._log("current", self.activation)

