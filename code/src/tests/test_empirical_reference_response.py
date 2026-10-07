from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.empirical_reference_response import EmpiricalReferenceResponse


def model():
    # A nonlinear interaction makes a mean-panel replacement an incorrect mean.
    return SimpleNamespace(fit_ids=['train_a', 'train_b', 'train_c'],
                           predict=lambda x: x.host.to_numpy() + x.host.to_numpy() * x.micro.to_numpy() ** 2)


def reference():
    return pd.DataFrame({'micro': [0., 2., 4.]}, index=['train_a', 'train_b', 'train_c'])


def test_nonlinear_observed_panel_mean_reconstruction_and_no_cap():
    wrapper = EmpiricalReferenceResponse(model(), reference(), ['micro'])
    query = pd.DataFrame({'host': [3., 10.], 'micro': [1., 4.]}, index=['x', 'y'])
    result = wrapper.components(query)
    np.testing.assert_allclose(result.reference_mean_response, query.host * (1 + 20 / 3))
    np.testing.assert_allclose(result.reference_mean_response + result.microbial_deviation, model().predict(query))
    assert result.microbial_deviation.abs().max() > 12
    panels = wrapper.reference_predictions(query.iloc[[0]])
    assert abs(np.mean(panels - result.reference_mean_response.iloc[0])) < 1e-12
    assert not np.isclose(result.reference_mean_response.iloc[0], 3 * (1 + reference().micro.mean() ** 2))


def test_batch_and_peer_independence_and_reference_copy():
    ref = reference()
    wrapper = EmpiricalReferenceResponse(model(), ref, ['micro'], batch_rows=3)
    query = pd.DataFrame({'host': [3., 10.], 'micro': [1., 4.]}, index=['x', 'y'])
    expected = wrapper.components(query)
    ref.loc[:, 'micro'] = 100
    np.testing.assert_array_equal(wrapper.components(query), expected)
    np.testing.assert_array_equal(wrapper.components(query.iloc[::-1]).iloc[::-1], expected)
    np.testing.assert_array_equal(wrapper.components(query.iloc[[0]]), expected.iloc[[0]])


def test_reference_participant_contract():
    ref = reference()
    ref.index = ['train_a', 'train_b', 'heldout']
    with pytest.raises(ValueError, match='training participants'):
        EmpiricalReferenceResponse(model(), ref, ['micro'])
    with pytest.raises(ValueError, match='one panel'):
        EmpiricalReferenceResponse(model(), pd.concat([reference(), reference()]), ['micro'])
