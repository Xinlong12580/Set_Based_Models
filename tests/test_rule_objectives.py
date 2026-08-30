import math
import pytest
from set_based_models import (
    FScoreRuleScore, LikelihoodRatioRuleScore, PurityCoverageRuleScore,
    RuleStatistics, SignificanceRuleScore, TPRFPRRuleScore, TargetFPRRuleScore,
)

def statistics():
    return RuleStatistics(40, 100, 10, 100, 40., 100., 10., 100., 2, 1)

def test_scores_have_expected_values():
    stats = statistics()
    assert TPRFPRRuleScore(lambda_fpr=2)(stats) == pytest.approx(.2)
    assert TargetFPRRuleScore(.05, 2)(stats) == pytest.approx(.3)
    assert PurityCoverageRuleScore()(stats) == pytest.approx(.32)
    assert FScoreRuleScore()(stats) == pytest.approx(2 * .8 * .4 / 1.2)
    assert SignificanceRuleScore()(stats) == pytest.approx(1600 / 50)
    assert LikelihoodRatioRuleScore()(stats) == pytest.approx(math.log(4), rel=1e-9)

@pytest.mark.parametrize("constructor", [
    lambda: TPRFPRRuleScore(lambda_fpr=-1),
    lambda: TargetFPRRuleScore(target_fpr=2, violation_penalty=1),
    lambda: FScoreRuleScore(beta=0),
    lambda: SignificanceRuleScore(background_uncertainty_fraction=-1),
    lambda: LikelihoodRatioRuleScore(background_smoothing=0),
])
def test_invalid_parameters_are_rejected(constructor):
    with pytest.raises(ValueError): constructor()
