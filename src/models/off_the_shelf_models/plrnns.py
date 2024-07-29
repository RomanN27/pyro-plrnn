from src.models.model_wrappers.cov_wrapper import create_covariance_wrapper
from src.models.initializers import npRNNInitialization, dendPLRNNInitialization
from src.models.normalizers import PLRNNMeanNormalizer
from src.models.cov_modules import ConstantCovariance, FixedCovariance
from src.models.transition_models.plrnns.plrnns import VanillaPLRNN, dendPLRNN, ClippedDendPLRNN

const_cov = create_covariance_wrapper(ConstantCovariance).get_decorator()
fixed_covariance = create_covariance_wrapper(FixedCovariance).get_decorator()
np_rnn_initializer = npRNNInitialization.get_decorator()
dend_plrnn_initializer = dendPLRNNInitialization().get_decorator()
mean_normalizer = PLRNNMeanNormalizer().get_decorator()

@const_cov
@np_rnn_initializer
class ConstCovVanillaPLRNN(VanillaPLRNN): ...


@const_cov
@dend_plrnn_initializer
@np_rnn_initializer
class ConstCovDendPLRNN(dendPLRNN): ...


@const_cov
@dend_plrnn_initializer
@np_rnn_initializer
class ConstCovClippedDendPLRNN(ClippedDendPLRNN): ...


@fixed_covariance
@np_rnn_initializer
class FixedCovVanillaPLRNN(VanillaPLRNN): ...


@fixed_covariance
@dend_plrnn_initializer
@np_rnn_initializer
class FixedCovDendPLRNN(dendPLRNN): ...


@const_cov
@mean_normalizer
@dend_plrnn_initializer
@np_rnn_initializer
class FixedMeanCenteredCovClippedDendPLRNN(ClippedDendPLRNN): ...

