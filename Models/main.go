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
	features, target, err := loadCSV("boston.csv")
	if err != nil {
		panic(err)
	}

	// Print the loaded data
	fmt.Println("Loaded Data:")
	for i, row := range features {
		fmt.Printf("Row %d: %v\n", i, row)
	}

	// Split the data into training and test sets (70% training, 30% test)
	trainSize := int(0.7 * float64(len(features)))
	trainFeatures := features[:trainSize]
	trainTarget := target[:trainSize]
	testFeatures := features[trainSize:]
	testTarget := target[trainSize:]

	// Add a constant term (intercept) to the features matrix
	trainFeaturesWithConstant := make([][]float64, len(trainFeatures))
	for i, row := range trainFeatures {
		trainFeaturesWithConstant[i] = append([]float64{1}, row...)
	}

	testFeaturesWithConstant := make([][]float64, len(testFeatures))
	for i, row := range testFeatures {
		testFeaturesWithConstant[i] = append([]float64{1}, row...)
	}

	// Set up to run 100 iterations
	numIterations := 100

	// Set the regularization parameter (lambda)
	lambda := 0.1

	// Set up channels to communicate the results from each Goroutine
	sumLinearCoefficients := make([]float64, len(trainFeatures[0])+1)
	sumRidgeCoefficients := make([]float64, len(trainFeatures[0])+1)

	// Perform linear regression for numIterations times
	for i := 0; i < numIterations; i++ {
		// Linear Regression
		linearCoefficients := linearRegression(trainFeatures, trainTarget)
		for j := range linearCoefficients {
			sumLinearCoefficients[j] += linearCoefficients[j]
		}
	}

	// Perform ridge regression for numIterations times
	for i := 0; i < numIterations; i++ {
		// Ridge Regression
		ridgeCoefficients := ridgeRegression(trainFeatures, trainTarget, lambda)
		for j := range ridgeCoefficients {
			sumRidgeCoefficients[j] += ridgeCoefficients[j]
		}
	}

	// Calculate the average coefficients for both linear and ridge
	avgLinearCoefficients := make([]float64, len(sumLinearCoefficients))
	avgRidgeCoefficients := make([]float64, len(sumRidgeCoefficients))
	for i := range sumLinearCoefficients {
		avgLinearCoefficients[i] = sumLinearCoefficients[i] / float64(numIterations)
		avgRidgeCoefficients[i] = sumRidgeCoefficients[i] / float64(numIterations)
	}

	// Variables to store the predicted home prices for each iteration
	numTestSamples := len(testFeatures)
	avgPredictedPricesLiner := make([]float64, numTestSamples)
	avgPredictedPricesRidge := make([]float64, numTestSamples)

	// Launch goroutines to run second machine learning model concurrently
	// Run the prediction 100 times
	for i := 0; i < numIterations; i++ {

		// Perform linear regression
		coefficientsLin := linearRegression(trainFeatures, trainTarget)

		// Make predictions on the test data
		predictionsLin := make([]float64, len(testFeatures))
		for j, row := range testFeatures {
			predictionsLin[j] = predictLin(row, coefficientsLin)
		}

		// Add the predicted home prices to the average for each feature
		for j, price := range predictionsLin {
			avgPredictedPricesLiner[j] += price / float64(numIterations)
		}

		// Perform Ridge Regression
		coefficientsRidge := ridgeRegression(trainFeatures, trainTarget, lambda)

		// Make predictions Ridge
		predictionsRidge := make([]float64, len(testFeatures))
		for j, row := range testFeatures {
			predictionsRidge[j] = predictRidge(row, coefficientsRidge)
		}

		// Add the predicted home prices to the average for each feature
		for j, price := range predictionsRidge {
			avgPredictedPricesRidge[j] += price / float64(numIterations)
		}

		// Take the log of the predictions Ridge
		for i := range avgPredictedPricesRidge {
			avgPredictedPricesRidge[i] = math.Log(avgPredictedPricesRidge[i])
		}
	}

	// Print the predicted home prices using linear regression
	fmt.Println("Average Predicted Home Prices using Linear Regression:")
	for _, price := range avgPredictedPricesLiner {
		fmt.Printf("%2f\n", price)
	}

	// Print the predicted home prices using ridge regression
	fmt.Println("Average Predicted Home Prices using Ridge Regression:")
	for _, price := range avgPredictedPricesRidge {
		fmt.Printf("%2f\n", price)
	}

	//Calculate and print the Mean Absolute Percentage Error (MAPE)
	mape := meanAbsolutePercentageError(avgPredictedPricesLiner, testTarget)
	fmt.Printf("Mean Absolute Percentage Error (MAPE): %.2f%%\n", mape)

	//Calculate and print the Mean Absolute Percentage Error (MAPE) Ridge
	mapeRidge := meanAbsolutePercentageError(avgPredictedPricesRidge, testTarget)
	fmt.Printf("Mean Absolute Percentage Error (MAPE) Ridge: %.2f%%\n", mapeRidge)

	// Calculate and print the Mean Squared Error (MSE)
	mse := meanSquaredError(avgPredictedPricesLiner, testTarget)
	fmt.Printf("Mean Squared Error (MSE): %.2f\n", mse)

	// Calculate and print the Mean Squared Error (MSE) Ridge
	mseRidge := meanSquaredError(avgPredictedPricesRidge, testTarget)
	fmt.Printf("Mean Squared Error (MSE): %.2f\n", mseRidge)

	// Calculate and print the Root Mean Squared Error (RMSE)
	rmse := rootMeanSquaredError(avgPredictedPricesLiner, testTarget)
	fmt.Printf("Root Mean Squared Error (RMSE): %.2f\n", rmse)

	// Calculate and print the Root Mean Squared Error (RMSE) Ridge
	rmseRidge := rootMeanSquaredError(avgPredictedPricesRidge, testTarget)
	fmt.Printf("Root Mean Squared Error (RMSE): %.2f\n", rmseRidge)

	// Calculate and print the Root Mean Squared Percentage Error (RMSPE)
	rmspe := rootMeanSquaredPercentageError(avgPredictedPricesLiner, testTarget)
	fmt.Printf("Root Mean Squared Percentage Error (RMSPE): %.2f%%\n", rmspe)

	// Calculate and print the Root Mean Squared Percentage Error (RMSPE) Ridge
	rmspeRidge := rootMeanSquaredPercentageError(avgPredictedPricesRidge, testTarget)
	fmt.Printf("Root Mean Squared Percentage Error (RMSPE): %.2f%%\n", rmspeRidge)

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
	for _, row := range features {
		fmt.Println(row)
	}
	fmt.Println("Loaded target:")
	fmt.Println(target)
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
	matFeatures := mat.NewDense(len(features), len(featuresWithConstant[0]), nil)
	matFeatures.Apply(func(i, j int, v float64) float64 { return featuresWithConstant[i][j] }, matFeatures)
	matTarget := mat.NewVecDense(len(target), target)
	coefficientsVec := mat.NewVecDense(len(coefficients), coefficients)

	// Compute the coefficients using linear regression
	regression := new(mat.Dense)
	regression.Solve(matFeatures, matTarget)
	coefficientsVec.CopyVec(regression.ColView(0))

	return coefficients
}

func ridgeRegression(features [][]float64, target []float64, lambda float64) []float64 {
	numFeatures := len(features[0])
	coefficients := make([]float64, len(features[0])+1)

	// Add a constant term (intercept) to the feature matrix
	featuresWithConstant := make([][]float64, len(features))
	for i, row := range features {
		featuresWithConstant[i] = append([]float64{1}, row...)
	}

	// Create the design matrix
	matFeatures := mat.NewDense(len(featuresWithConstant), len(featuresWithConstant[0]), nil)
	matFeatures.Apply(func(i, j int, v float64) float64 { return featuresWithConstant[i][j] }, matFeatures)

	// Create the target vector
	matTarget := mat.NewVecDense(len(target), target)

	// Compute X^T * X and X^T * y
	var xtX mat.Dense
	xtX.Mul(matFeatures.T(), matFeatures)

	var xtY mat.VecDense
	xtY.MulVec(matFeatures.T(), matTarget)

	// Create the identity matrix
	identity := mat.NewDense(numFeatures+1, numFeatures+1, nil)
	for i := 0; i < len(featuresWithConstant[0]); i++ {
		identity.Set(i, i, lambda)
	}

	// Add 1 to the diagonal elements of the identity matrix
	for i := 1; i < numFeatures+1; i++ {
		identity.Set(i, i, 1.0+lambda)
	}

	// Compute the coefficients using ridge regression formula: inv(X^T * X + λI) * X^T * y
	var inv mat.Dense
	inv.Inverse(identity)

	var ridgeCoefficients mat.VecDense
	ridgeCoefficients.MulVec(&inv, &xtY)

	// Convert the ridgeCoefficients to a flat slice
	for i := 0; i < numFeatures+1; i++ {
		coefficients[i] = ridgeCoefficients.At(i, 0)
	}

	return coefficients
}

func predictLin(featureRow []float64, coefficients []float64) float64 {
	// Add constant term (intercept) to the feature row
	featureRowWithConstant := append([]float64{1}, featureRow...)
	if len(featureRowWithConstant) != len(coefficients) {
		panic("Feature row and coefficients length mismatch")
	}

	return floats.Dot(featureRowWithConstant, coefficients)
}

func predictRidge(featureRow []float64, coefficientsRidge []float64) float64 {
	// Add a constant term (intercept) to the feature row
	featureRowRidge := append([]float64{1}, featureRow...)
	if len(featureRowRidge) != len(coefficientsRidge) {
		panic("Feature row and coefficients length mismatch")
	}

	return floats.Dot(featureRowRidge, coefficientsRidge)
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
