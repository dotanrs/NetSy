import random
import math

DEFAULT_CONNECTION_STRENGTH = 10
ACTIVATION_RANGE_HIGH = 2
ACTIVATION_RANGE_LOW = -1.5
DEFAULT_NOISE_MEAN = 0.5
DEFAULT_DECAY = 0.1
SPIKE_SIZE = 3
DEFAULT_THRESHOLD_P = 0.05

def get_from_kwargs_or_default(argname, default, **kwargs):
	return kwargs[argname] if argname in kwargs else default


class Utils:

	def decay(var, coef):
		return var - var * coef


def neuron_factory(ntype, **kwargs):
	if ntype == "whitenoise":
		return WhiteNoiseNeuron(**kwargs)

	elif ntype == "threshold":
		return ThresholdNeuron(**kwargs)

	elif ntype == "binarynoise":
		return BinaryNoiseNeuron(**kwargs)

	raise(KeyError("no such neuron type"))


class Neuron:

	def __init__(self, name = "anonymous neuron", **kwargs):
		range_default = [ACTIVATION_RANGE_LOW, ACTIVATION_RANGE_HIGH]
		self.range = get_from_kwargs_or_default("range", range_default, **kwargs)
		self.lifespan = get_from_kwargs_or_default("lifespan", False, **kwargs)
		self.log = get_from_kwargs_or_default("log", False, **kwargs)
		self.refractory_time = get_from_kwargs_or_default("refractory_time", 2, **kwargs)

		self.out_synapses = []
		self.activation = self.range[0]
		self.name = name
		self.in_synapses = 0
		self.input = 0
		self.ticks = 0
		self.in_refractory_period = False
		self.refractory_period_timer = 0

	def _log(self, param, value):
		if not self.log:
			return
		print("[%s] %s = %s" % (self.name, param, str(value)))

	def _apply_input(self):
		self.activation += self.input
		self.input = 0

	def _apply_range(self):
		if self.activation > self.range[1]:
			self.activation = self.range[1]

		if self.activation < self.range[0]:
			self.activation = self.range[0]

	def _calc_output(self):
		self._apply_input()
		self._apply_range()

	def _post_transmission(self):
		self.activation = 0

	def run(self):

		if self.lifespan and self.ticks > self.lifespan:
			self.activation = 0
			for neuron, connection_strength in self.out_synapses:
				neuron._decrease_input_count(1)
			return
			
		self._post_transmission()

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

			self._log("Sengind signal", signal)

			for neuron, connection_strength in self.out_synapses:
				neuron._log("Receiving signal", str(signal) + "*" + str(connection_strength))
				neuron.recieve_signal(signal * connection_strength)

		self.ticks += 1

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

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		default_threshold = self.range[1]
		self.threshold 		   = get_from_kwargs_or_default("threshold", 1, **kwargs)
		self.decay_coefficient = get_from_kwargs_or_default("decay_coefficient", DEFAULT_DECAY, **kwargs)
		self.range[1] = SPIKE_SIZE
		self.name = "[TH] " + self.name
		self._log("range", self.range)

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
			

	def _post_transmission(self):
		if self.activation >= SPIKE_SIZE:
			self.in_refractory_period = True
			self.refractory_period_timer = 1
			self.activation = self.range[0]

		self.activation = Utils.decay(self.activation, self.decay_coefficient)


class WhiteNoiseNeuron(Neuron):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.mean = get_from_kwargs_or_default("mean", DEFAULT_NOISE_MEAN, **kwargs)
		self.name = "[WN] " + self.name

	def get_signal(self):
		self.activation = random.random() * self.mean * 2
		return self.activation

	def _post_transmission(self):
		pass

class BinaryNoiseNeuron(ThresholdNeuron):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.p = get_from_kwargs_or_default("p", DEFAULT_THRESHOLD_P, **kwargs)
		self.name = "[BN] " + self.name

	def _apply_input(self):
		pass

	def get_signal(self):
		signal = random.random()
		if signal < self.p :
			self.activation = self.range[1]
		else:
			self.activation = self.range[0]
			
		return self.activation

