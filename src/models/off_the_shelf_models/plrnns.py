from src.models.model_wrappers.cov_wrapper import create_covariance_wrapper
from src.models.initializer_mixins import NormalizedPositiveDefiniteInitialization, dendPLRNNInitialization
from src.models.normalizer_mixins import PLRNNMeanNormalizer
from src.models.cov_modules import ConstantCovariance, FixedCovariance
from src.models.transition_models.plrnns.raw_plrnns import _VanillaPLRNN, _DendPLRNN, _ClippedDendPLRNN

const_cov = create_covariance_wrapper(ConstantCovariance).get_decorator()
fixed_covariance = create_covariance_wrapper(FixedCovariance).get_decorator()
np_rnn_initializer = NormalizedPositiveDefiniteInitialization.get_decorator()
dend_plrnn_initializer = dendPLRNNInitialization().get_decorator()
mean_normalizer = PLRNNMeanNormalizer().get_decorator()

@const_cov
@np_rnn_initializer
class ConstCovVanillaPLRNN(_VanillaPLRNN): ...


@const_cov
@dend_plrnn_initializer
@np_rnn_initializer
class ConstCovDendPLRNN(_DendPLRNN): ...


@const_cov
@dend_plrnn_initializer
@np_rnn_initializer
class ConstCovClippedDendPLRNN(_ClippedDendPLRNN): ...


@fixed_covariance
@np_rnn_initializer
class FixedCovVanillaPLRNN(_VanillaPLRNN): ...


@fixed_covariance
@dend_plrnn_initializer
@np_rnn_initializer
class FixedCovDendPLRNN(_DendPLRNN): ...


@const_cov
@mean_normalizer
@dend_plrnn_initializer
@np_rnn_initializer
class FixedMeanCenteredCovClippedDendPLRNN(_ClippedDendPLRNN): ...

