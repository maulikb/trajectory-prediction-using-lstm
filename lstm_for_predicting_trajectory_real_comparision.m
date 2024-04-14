%% Data Pre-Processing

% Load data from CSV file
data = readtable('Autonoumous_Car_Data.csv'); % Ensure file path and name are correct

% Normalize features
features = data{:, {'Latitude', 'Longitude', 'heading', 'v'}};
[features, featureMu, featureSigma] = zscore(features);

% Define target variable and normalize if needed
sequenceLength = 10; % Example sequence length
numFeatures = size(features, 2);
numSamples = size(features, 1) - sequenceLength + 1;

% Create input-output pairs with sequences
X = zeros(numSamples, sequenceLength, numFeatures);
Y = zeros(numSamples, 1); % Modify as per your target structure
for i = 1:numSamples
    X(i, :, :) = features(i:i+sequenceLength-1, :);
    Y(i) = features(i+sequenceLength-1, 1); % Predicting next Latitude as an example
end

% Split data into training, validation, and test sets
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

numTrainSamples = floor(trainRatio * numSamples);
numValSamples = floor(valRatio * numSamples);
numTestSamples = numSamples - numTrainSamples - numValSamples;

XTrain = X(1:numTrainSamples, :, :);
YTrain = Y(1:numTrainSamples);
XVal = X(numTrainSamples+1:numTrainSamples+numValSamples, :, :);
YVal = Y(numTrainSamples+1:numTrainSamples+numValSamples);
XTest = X(numTrainSamples+numValSamples+1:end, :, :);
YTest = Y(numTrainSamples+numValSamples+1:end);

% Convert datasets to cell arrays for LSTM input
XTrainCell = num2cell(XTrain, [2,3]);
XValCell = num2cell(XVal, [2,3]);
XTestCell = num2cell(XTest, [2,3]);

%% LSTM Network Architecture

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50, 'OutputMode', 'sequence')
    lstmLayer(50, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XValCell, YVal}, ...
    'ValidationFrequency', 10, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

%% Training the Model

model = trainNetwork(XTrainCell, YTrain, layers, options);

%% Model Evaluation

% Predict on the test set
YPred = predict(model, XTestCell);

% Calculate Root Mean Squared Error (RMSE)
rmse = sqrt(mean((YPred - YTest).^2));
fprintf('Root Mean Squared Error (RMSE) on test set: %.4f\n', rmse);

%% Plot the Actual vs. Predicted Values

figure;
plot(1:length(YTest), YTest, '-o', 'DisplayName', 'Actual');
hold on;
plot(1:length(YPred), YPred, '-o', 'DisplayName', 'Predicted');
xlabel('Sample Index');
ylabel('Normalized Latitude');
title('Comparison of Actual and Predicted Values');
legend show;
grid on;
