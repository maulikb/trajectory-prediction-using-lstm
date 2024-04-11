%% Data Pre-Processing

% Load data from CSV file
data = readtable('Autonoumous_Car_Data.csv');

% Calculate distance between consecutive points using Haversine formula
R = 6371; % Earth's radius in kilometers
for i = 1:(height(data)-1)
    lat1 = deg2rad(data.Latitude(i));
    lat2 = deg2rad(data.Latitude(i+1));
    lon1 = deg2rad(data.Longitude(i));
    lon2 = deg2rad(data.Longitude(i+1));
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    data.Distance(i) = R * c;
end

% Normalize features and target
features = data{:, {'Latitude', 'Longitude', 'heading', 'v'}};
target = data.Distance;
[features, featureMu, featureSigma] = zscore(features);
target = (target - mean(target)) / std(target);

% Create input-output pairs with sequences
sequenceLength = 10; % Example sequence length
numFeatures = size(features, 2);
numSamples = size(features, 1) - sequenceLength + 1;
X = zeros(numSamples, sequenceLength, numFeatures);
Y = zeros(numSamples, 1);
for i = 1:numSamples
    X(i, :, :) = features(i:i+sequenceLength-1, :);
    Y(i) = target(i+sequenceLength-1);
end


disp(size(features));  % Should be [numSamples, 4]
disp(size(X));         % Should be [numSamples - sequenceLength + 1, sequenceLength, 4]
disp(size(Y));         % Should be [numSamples - sequenceLength + 1, 1]

%% Data Formulation

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


% For XVal
XValCell = cell(size(XVal, 1), 1);
for i = 1:size(XVal, 1)
    XValCell{i} = squeeze(XVal(i, :, :))';
end

% For XTest
XTestCell = cell(size(XTest, 1), 1);
for i = 1:size(XTest, 1)
    XTestCell{i} = squeeze(XTest(i, :, :))';
end


%% LSTM Network Architecture
% Define LSTM network architecture
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50, 'OutputMode', 'sequence')
    lstmLayer(50, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer];

% Compile the model
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XValCell, YVal}, ...
    'ValidationFrequency', 10, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');


%% Training
disp(size(XTrain));  % Should be [numTrainSamples, sequenceLength, numFeatures]
disp(size(YTrain));  % Should be [numTrainSamples, 1]
disp(class(XTrain));  % Should be 'double' or 'single'
disp(class(YTrain));
disp(layers);

if any(isnan(XTrain), 'all') || any(isnan(YTrain), 'all')
    error('Training data contains NaN values.');
end
if any(isinf(XTrain), 'all') || any(isinf(YTrain), 'all')
    error('Training data contains Inf values.');
end


% Initialize the cell array
XTrainCell = cell(size(XTrain, 1), 1);

% Convert each sequence to the required format
for i = 1:size(XTrain, 1)
    XTrainCell{i} = squeeze(XTrain(i, :, :))';
end

model = trainNetwork(XTrainCell, YTrain, layers, options);



%% Evaluation


% Evaluate the trained LSTM model on the test set
YPred = predict(model, XTestCell);
rmse = sqrt(mean((YPred - YTest).^2));
fprintf('Root Mean Squared Error (RMSE) on test set: %.4f\n', rmse);

