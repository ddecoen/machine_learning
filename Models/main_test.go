package main

import (
	"testing"
)

func TestLinearRegression(t *testing.T) {
	// Create sample data for testing linearRegression
	features := [][]float64{
		{1, 2},
		{2, 3},
		{3, 4},
	}
	target := []float64{3, 4, 5}

	// Call the linearRegression function
	coefficients := linearRegression(features, target)

	// Perform assertions on the coefficients
	expectedLength := len(features[0]) + 1 // Expected length of the coefficient vector (including the intercept)
	if len(coefficients) != expectedLength {
		t.Errorf("Unexpected coefficient vector length. Expected %d, got %d", expectedLength, len(coefficients))
	}
}

func TestRidgeRegression(t *testing.T) {
	// Create sample data for testing ridgeRegression
	features := [][]float64{
		{1, 2},
		{2, 3},
		{3, 4},
	}
	target := []float64{3, 4, 5}
	lambda := 0.1

	// Call the ridgeRegression function
	coefficients := ridgeRegression(features, target, lambda)

	// Perform assertions on the coefficients
	expectedLength := len(features[0]) + 1 // Expected length of the coefficient vector (including the intercept)
	if len(coefficients) != expectedLength {
		t.Errorf("Unexpected coefficient vector length. Expected %d, got %d", expectedLength, len(coefficients))
	}
}

func TestPredictLin(t *testing.T) {
	// Create sample data for testing predictLin
	featureRow := []float64{2, 3}
	coefficients := []float64{-1, 2, 1} // Assuming a simple linear regression y = -1 + 2*x1 + 1*x2

	// Call the predictLin function with the provided coefficients
	prediction := predictLin(featureRow, coefficients)

	// Perform assertions on the prediction or other tests as needed
	// For example, you can check if the prediction has the expected value:
	expectedPrediction := 6.0 // Assuming a simple linear regression y = -1 + 2*2 + 1*3
	if prediction != expectedPrediction {
		t.Errorf("Unexpected prediction. Expected %f, got %f", expectedPrediction, prediction)
	}
}
