package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	startTime := time.Now()
	// Load the data from a CSV file (or any other data source)
	features, target, err := loadCSV("../boston.csv")
	if err != nil {
		panic(err)
	}

	// Print the loaded data
	fmt.Println("Loaded Data:")
	for _, row := range features {
		fmt.Println(row)
	}

	// Split the data into training and test sets (70% training, 30% test)
	trainSize := int(0.7 * float64(len(features)))
	trainFeatures := features[:trainSize]
	trainTarget := target[:trainSize]
	testFeatures := features[trainSize:]
	testTarget := target[trainSize:]

	// Set up to run 100 iterations
	numIterations := 100

	// Variables to store the predicted home prices for each iteration
	avgPredictedPrices := make([]float64, len(testFeatures))

	// Launch goroutines to run second machine learning model concurrently
	// Run the prediction 100 times
	for i := 0; i < numIterations; i++ {
		// Perform linear regression
		coefficients := linearRegression(trainFeatures, trainTarget)

		// Make predictions on the test data
		predictions := make([]float64, len(testFeatures))
		for i, row := range testFeatures {
			predictions[i] = predict(row, coefficients)
		}

		// Add the predicted home prices to the average for each feature
		for j, price := range predictions {
			avgPredictedPrices[j] += price / float64(numIterations)
		}

	}

	// Print the predicted home prices
	fmt.Println("Average Predicted Home Prices:")
	for _, price := range avgPredictedPrices {
		fmt.Printf("%2f\n", price)
	}

	//Calculate and print the Mean Absolute Percentage Error (MAPE)
	mape := meanAbsolutePercentageError(avgPredictedPrices, testTarget)
	fmt.Printf("Mean Absolute Percentage Error (MAPE): %.2f%%\n", mape)

	// Calculate and print the Mean Squared Error (MSE)
	mse := meanSquaredError(avgPredictedPrices, testTarget)
	fmt.Printf("Mean Squared Error (MSE): %.2f\n", mse)

	// Calculate and print the Root Mean Squared Error (RMSE)
	rmse := rootMeanSquaredError(avgPredictedPrices, testTarget)
	fmt.Printf("Root Mean Squared Error (RMSE): %.2f\n", rmse)

	// Calculate and print the Root Mean Squared Percentage Error (RMSPE)
	rmspe := rootMeanSquaredPercentageError(avgPredictedPrices, testTarget)
	fmt.Printf("Root Mean Squared Percentage Error (RMSPE): %.2f%%\n", rmspe)

	// Calculate the time to run func main
	duration := time.Since(startTime)
	fmt.Printf("Time to execute code: %s\n", duration)

}

func loadCSV(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	lines, err := parseCSV(file)
	if err != nil {
		return nil, nil, err
	}

	// Skip the first row (header) and prepare slices to store features and target
	features := make([][]float64, len(lines)-1)
	target := make([]float64, len(lines)-1)

	for i, line := range lines[1:] {
		features[i] = make([]float64, len(line)-2)  // Skip the first and last columns
		for j, val := range line[1 : len(line)-1] { // Skip the first and last columns
			features[i][j], err = strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, nil, err
			}
		}
		target[i], err = strconv.ParseFloat(line[len(line)-1], 64) // Use the last column as the target
		if err != nil {
			return nil, nil, err
		}
	}

	// Check data
	fmt.Println("Loaded features:")
	fmt.Printf("%2f\n", features)
	fmt.Println("Loaded target:")
	fmt.Printf("%2f\n", target)
	return features, target, nil

}

func parseCSV(file *os.File) ([][]string, error) {
	lines := make([][]string, 0)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), ",")
		lines = append(lines, line)
	}
	return lines, scanner.Err()
}

func linearRegression(features [][]float64, target []float64) []float64 {
	coefficients := make([]float64, len(features[0])+1)

	// Add a constant term (intercept) to the features matrix
	featuresWithConstant := make([][]float64, len(features))
	for i, row := range features {
		featuresWithConstant[i] = append([]float64{1}, row...)
	}

	// Perform linear regression to calculate the coefficients
	matFeatures := mat.NewDense(len(featuresWithConstant), len(featuresWithConstant[0]), nil)
	matFeatures.Apply(func(i, j int, v float64) float64 { return featuresWithConstant[i][j] }, matFeatures)
	matTarget := mat.NewVecDense(len(target), target)
	coefficientsVec := mat.NewVecDense(len(coefficients), coefficients)

	// Compute the coefficients using linear regression
	regression := new(mat.Dense)
	regression.Solve(matFeatures, matTarget)
	coefficientsVec.CopyVec(regression.ColView(0))

	return coefficients
}

func predict(featureRow []float64, coefficients []float64) float64 {
	if len(featureRow)+1 != len(coefficients) {
		panic("Feature row and coefficients length mismatch")
	}

	// Add a constant term (intercept) to the feature row
	featureRowWithConstant := append([]float64{1}, featureRow...)
	return floats.Dot(featureRowWithConstant, coefficients)
}

func meanAbsolutePercentageError(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		panic("Predictions and targets length mismatch")
	}

	var sumPercentageError float64
	for i, pred := range predictions {
		percentageError := math.Abs((pred - targets[i]) / targets[i])
		sumPercentageError += percentageError
	}

	meanPercentageError := sumPercentageError / float64(len(predictions))
	return meanPercentageError
}

func meanSquaredError(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		panic("Predictions and targets length mismatch")
	}

	var sumSquaredError float64
	for i, pred := range predictions {
		diff := pred - targets[i]
		sumSquaredError += diff * diff
	}

	return sumSquaredError / float64(len(predictions))
}

func rootMeanSquaredError(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		panic("Predictions and targets length mismatch")
	}

	var sumSquaredError float64
	for i, pred := range predictions {
		diff := pred - targets[i]
		sumSquaredError += diff * diff
	}

	meanSquaredError := sumSquaredError / float64(len(predictions))
	return math.Sqrt(meanSquaredError)

}

func rootMeanSquaredPercentageError(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		panic("Predictions and targets length mismatch")
	}

	var sumSquaredPercentageError float64
	for i, pred := range predictions {
		percentageError := (pred - targets[i]) / targets[i]
		sumSquaredPercentageError += percentageError * percentageError
	}

	meanSquaredPercentageError := sumSquaredPercentageError / float64(len(predictions))
	return math.Sqrt(meanSquaredPercentageError)

}
