% current file's directory
currentDir = fileparts(mfilename('fullpath'));

% relative path to the dataset
datasetBasePath = fullfile(currentDir, '..', 'data', 'Dataset');


% disease categories based on folder names
diseaseCategories = {'curl_stage1', 'curl_stage1+curl_stage2+sooty', 'curl_stage1+sooty', ...
                     'curl_stage2', 'curl_stage2+sooty', 'healthy'};

% Initialize feature and label arrays
features = [];
labels = [];

% Loop over each category
for i = 1:length(diseaseCategories)
    % Define the full path to the folder
    folderPath = fullfile(datasetBasePath, diseaseCategories{i});

    % List all JPEG and PNG images in the folder
    imageFilesJPG = dir(fullfile(folderPath, '*.jpg'));
    imageFilesPNG = dir(fullfile(folderPath, '*.png'));
    imageFiles = [imageFilesJPG; imageFilesPNG]; % Combine file lists

    % Process each image
    for j = 1:length(imageFiles)
        % Load image
        img = imread(fullfile(folderPath, imageFiles(j).name));

        % Preprocess the image
        % Convert to grayscale
        img_gray = rgb2gray(img);

        % Apply Gaussian filter to smooth the image
        h = fspecial('gaussian', [5 5], 2);
        img_smooth = imfilter(img_gray, h, 'replicate');

        % Resize image to standard dimensions (e.g., 256x256 pixels)
        img_resized = imresize(img_smooth, [256 256]);

        % Feature extraction
        % Edge detection using Canny
        edges = edge(img_resized, 'canny');

        % Color histogram
        [hist_counts, ~] = imhist(img_resized);

        % Image segmentation using Otsu's method
        threshold = graythresh(img_resized);
        img_segmented = imbinarize(img_resized, threshold);

        % Flatten the edge, histogram, and segmented image features into feature vectors
        edge_features = edges(:)';
        hist_features = hist_counts';
        seg_features = img_segmented(:)';

        % Combine all extracted features into one feature vector per image
        img_features = [edge_features, hist_features, seg_features];

        % Append the extracted features and label to the dataset
        features = [features; img_features];
        labels = [labels; i]; % Assign a numeric label corresponding to the disease category
    end
end

% Define the ratio of the dataset to be used for training and testing
trainRatio = 0.8; % 80% of the data for training
testRatio = 0.2; % 20% of the data for testing

% Calculate the number of samples for each set
numSamples = size(features, 1);
numTrain = round(trainRatio * numSamples);
numTest = numSamples - numTrain;

% Randomly shuffle the data and the corresponding labels
randIndices = randperm(numSamples);
shuffledFeatures = features(randIndices, :);
shuffledLabels = labels(randIndices);

trainFeatures = [];
trainLabels = [];
testFeatures = [];
testLabels = [];

trainRatio = 0.8; % 80% for training

for i = 1:length(diseaseCategories)
    % Indices for all samples from category i
    catIndices = find(labels == i);
    numCatSamples = length(catIndices);
    numTrainSamples = round(trainRatio * numCatSamples);
    
    % Shuffle indices for category i
    catIndices = catIndices(randperm(numCatSamples));
    
    % Split into train and test for category i
    trainCatIndices = catIndices(1:numTrainSamples);
    testCatIndices = catIndices(numTrainSamples+1:end);
    
    % Add to the overall train/test sets
    trainFeatures = [trainFeatures; features(trainCatIndices, :)];
    trainLabels = [trainLabels; labels(trainCatIndices)];
    testFeatures = [testFeatures; features(testCatIndices, :)];
    testLabels = [testLabels; labels(testCatIndices)];
end

% KNN model using cross-validation to find the best number of neighbors
bestK = 1;
bestAccuracy = 0;

for k = 1:2:15 % Test odd values of k to avoid ties
    cvModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', k, 'CrossVal', 'on', 'KFold', 5);
    classLoss = kfoldLoss(cvModel, 'LossFun', 'ClassifError');
    if (1 - classLoss) > bestAccuracy
        bestAccuracy = 1 - classLoss;
        bestK = k;
    end
end

% Train the final model using the bestK value
knnModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', bestK);

% Predict labels for the test set
predictedLabels = predict(knnModel, testFeatures);

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

%confusion matrix for results
confMat = confusionmat(testLabels, predictedLabels);

% Safe calculation of precision, recall, and F1 score
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
precision(isnan(precision)) = 0; % Replace NaN with 0
recall(isnan(recall)) = 0; % Replace NaN with 0
f1Scores = 2 * (precision .* recall) ./ (precision + recall);
f1Scores(isnan(f1Scores)) = 0; % Replace NaN with 0 for classes with no predictions

% print out the performance metrics
fprintf('Optimal K: %d\n', bestK);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall: %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(f1Scores));

% visual for the confusion matrix
confusionchart(confMat, diseaseCategories)