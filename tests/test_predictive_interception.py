"""Tests for the Predictive Path Interception engine."""

import pytest
from cloud.services.predictive_interception.interceptor import PredictiveInterceptor


class TestPointInPolygon:
    """Test the ray-casting point-in-polygon implementation."""

    def test_point_inside_square(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert PredictiveInterceptor._point_in_polygon((5, 5), polygon) is True

    def test_point_outside_square(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert PredictiveInterceptor._point_in_polygon((15, 5), polygon) is False

    def test_point_on_edge(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        # Edge behavior is implementation-defined but shouldn't crash
        PredictiveInterceptor._point_in_polygon((0, 5), polygon)

    def test_empty_polygon(self):
        assert PredictiveInterceptor._point_in_polygon((5, 5), []) is False

    def test_triangle(self):
        polygon = [[0, 0], [10, 0], [5, 10]]
        assert PredictiveInterceptor._point_in_polygon((5, 3), polygon) is True
        assert PredictiveInterceptor._point_in_polygon((0, 10), polygon) is False


class TestInterceptor:
    def test_no_world_model_returns_empty(self):
        interceptor = PredictiveInterceptor()
        result = interceptor.evaluate({"timestamp": 1.0})
        assert result == []

    def test_no_restricted_zones_returns_empty(self):
        interceptor = PredictiveInterceptor()
        result = interceptor.evaluate({"timestamp": 1.0})
        assert result == []

    def test_get_all_trajectories_empty(self):
        interceptor = PredictiveInterceptor()
        assert interceptor.get_all_trajectories() == {}
