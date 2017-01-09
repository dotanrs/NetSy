import random
import math

DEFAULT_CONNECTION_STRENGTH = 10
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

	def __init__(self, name = "anonymous neuron", **kwargs):
		range_default = [ACTIVATION_RANGE_LOW, ACTIVATION_RANGE_HIGH]

		# The minimal and maximal values of activation the neuron can have
		self.range = get_from_kwargs_or_default("range", range_default, **kwargs)

		# If set, the neuron will day after this many iterations
		self.lifespan = get_from_kwargs_or_default("lifespan", False, **kwargs)

		# If set, the neuron will print logs to console
		self.log = get_from_kwargs_or_default("log", False, **kwargs)

		# The length of the refractory period, if set
		self.refractory_time = get_from_kwargs_or_default("refractory_time", 2, **kwargs)

		# Array of neurons to send the signal to
		self.out_synapses = []

		# Number of iterations passed
		self.ticks = 0

		self.activation = self.range[0]
		self.name = self._name_prefix + name
		self.in_synapses = 0
		self.input = 0
		self.in_refractory_period = False
		self.refractory_period_timer = 0
		self.is_active = True

	@property
	def _name_prefix(self):
		return "[{0}] ".format(self._type_identifier())

	def _type_identifier(self):
		return "N"

	def _log(self, param, value):
		if not self.log:
			return
		print("[%s] %s = %s" % (self.name, param, str(value)))

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
			return

		self.ticks += 1

		if self.lifespan and self.ticks > self.lifespan:
			self.activation = 0
			self.is_active = False
			for neuron, connection_strength in self.out_synapses:
				neuron._decrease_input_count(1)
			return
		
		self._internal_processes()

		if self.in_refractory_period:
			if self.refractory_period_timer > self.refractory_time:
				self.in_refractory_period = False
			else:
				self.refractory_period_timer += 1
				self.activation = self.range[0]
				return

		self._calc_output()

		signal = self.get_signal()

		if signal !=0:
			self._log("Sendind signal", signal)
			for neuron, connection_strength in self.out_synapses:
				# neuron._log("Receiving signal", str(signal) + " * " + str(connection_strength))
				neuron.recieve_signal(signal * connection_strength)


	def show_log(self):
		self.log = True

	def get_name(self):
		return self.name

	def get_activation(self):
		return self.activation

	def get_signal(self):
		return self.activation

	def recieve_signal(self, signal):
		self.input += signal

	def _increase_input_count(self, by=1):
		self.in_synapses += by

	def _decrease_input_count(self, by=1):
		self.in_synapses = max(self.in_synapses - by, 1)

	def _append_to_synapses(self, neuron, connection_strength = DEFAULT_CONNECTION_STRENGTH):
		self.out_synapses.append((neuron, connection_strength))

	def send_to(self, neuron, connection_strength = DEFAULT_CONNECTION_STRENGTH):
		if isinstance(neuron, list):
			for single_neuron in neuron:
				self.send_to(single_neuron, connection_strength)
			return

		self._append_to_synapses(neuron, connection_strength)
		neuron._increase_input_count()

	def listen_to(self, neuron, connection_strength = DEFAULT_CONNECTION_STRENGTH):
		if isinstance(neuron, list):
			for single_neuron in neuron:
				self.listen_to(single_neuron, connection_strength)
			return

		if not (isinstance(connection_strength, int) or isinstance(connection_strength, float)):
			connection_strength = random.random()

		neuron._append_to_synapses(self, connection_strength)
		self._increase_input_count()

	def set_lifespan(self, new_value):
		self.lifespan = new_value

	# TODO
	def update_connection_strength(self, neuron, value):
		pass


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

		if self.in_synapses > 0 and not self.in_refractory_period:
			self.activation += self.input # / self.in_synapses
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

	def _type_identifier(self):
		return "SIG"

	def _apply_range(self):
		pass

	def get_signal(self):
		if self.in_synapses:
			self.activation /= self.in_synapses
		else:
			return 0
		return math.tanh(self.activation) * self.mean

	def _internal_processes(self):
		self.activation = 0

