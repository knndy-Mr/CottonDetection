  % Define dataset base path
  datasetBasePath = '/Users/kkwillijr/Documents/Image-Processing/Dataset';
  
  % Define disease categories based on folder names
  diseaseCategories = {'curl_stage1', 'curl_stage1+curl_stage2+sooty', 'curl_stage1+sooty', ...
                       'curl_stage2', 'curl_stage2+sooty', 'healthy'};
  
  % Initialize feature and label arrays
  features = [];
  labels = [];
  
  % Loop over each category to populate 'features' and 'labels'
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
  
          % Resize image to standard dimensions (e.g., 256x256 pixels)
          img_resized = imresize(img_gray, [256 256]);
  
          % Apply Gaussian filter to smooth the image
          % (This could be an optional step based on your preference)
          h = fspecial('gaussian', [5 5], 2);
          img_smooth = imfilter(img_resized, h, 'replicate');
  
          % Edge features
          edges = edge(img_smooth, 'canny');
          % Consider summarizing edge features, e.g., by counting the edges or using a more complex descriptor
          edge_summary = sum(edges(:));  % This summarizes the edges into a single count
      
          % GLCM texture features
          % Calculate texture features from GLCM
          glcm = graycomatrix(img_smooth, 'Offset', offsets);
          stats = graycoprops(glcm, {'contrast', 'homogeneity', 'correlation', 'energy'});
          % Flatten texture features into a 1D vector and summarize if necessary
          texture_features = [stats.Contrast, stats.Homogeneity, stats.Correlation, stats.Energy];
      
          % Ensure texture_features is a 1D vector with a consistent number of elements
          assert(isvector(texture_features), 'Texture features must be a 1D vector');
      
          % Color histogram features
          
          % Combine all features into a single vector
          img_features = [edge_summary, texture_features, color_features];
      
          % Append the features and label
          features = [features; img_features];
          labels = [labels; i];
      end
  end
  
  % Stratified split into training and test sets% Define the ratio for splitting
  trainRatio = 0.8; % 80% for training
  
  % Initialize stratified train/test features and labels
  trainFeatures = [];
  trainLabels = [];
  testFeatures = [];
  testLabels = [];
  
  for i = 1:length(diseaseCategories)
      % Indices for all samples from category i
      catIndices = find(labels == i);
      numCatSamples = length(catIndices);
      numTrainSamples = round(trainRatio * numCatSamples);
      
      % Shuffle indices for category i
      shuffledCatIndices = catIndices(randperm(numCatSamples));
      
      % Split into train and test for category i
      trainCatIndices = shuffledCatIndices(1:numTrainSamples);
      testCatIndices = shuffledCatIndices(numTrainSamples+1:end);
      
      % Add to the overall train/test sets
      trainFeatures = [trainFeatures; features(trainCatIndices, :)];
      trainLabels = [trainLabels; labels(trainCatIndices)];
      testFeatures = [testFeatures; features(testCatIndices, :)];
      testLabels = [testLabels; labels(testCatIndices)];
  end
  
  
  % Create and train the KNN model using cross-validation to find the best number of neighbors
  % Initialize cross-validation settings
  bestK = 1;
  bestAccuracy = 0;
  
  % Cross-validation to find the best 'NumNeighbors' for KNN
  for k = 1:2:15 % Testing odd values of k to avoid ties
      cvModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', k, 'CrossVal', 'on', 'KFold', 5);
      classLoss = kfoldLoss(cvModel, 'LossFun', 'ClassifError');
      cvAccuracy = 1 - classLoss;
      if cvAccuracy > bestAccuracy
          bestAccuracy = cvAccuracy;
          bestK = k;
      end
  end
  
  
  % Train the final model using the bestK value
  finalKnnModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', bestK);
  
  % Predict labels for the test set and calculate accuracy
  % Predict labels for the test set
  predictedLabels = predict(finalKnnModel, testFeatures);
  
  % Calculate accuracy
  accuracy = sum(predictedLabels == testLabels) / length(testLabels);
  
  % Create a confusion matrix
  confMat = confusionmat(testLabels, predictedLabels);
  
  % Calculation of precision, recall, and F1 score, safely handling division by zero
  precision = diag(confMat) ./ sum(confMat, 2);
  precision(isnan(precision)) = 0; % Replace NaN with 0
  recall = diag(confMat) ./ sum(confMat, 1)';
  recall(isnan(recall)) = 0; % Replace NaN with 0
  f1Scores = 2 * (precision .* recall) ./ (precision + recall);
  f1Scores(isnan(f1Scores)) = 0; % Replace NaN with 0 for classes with no predictions
  
  % Print out the performance metrics
  fprintf('Optimal K: %d\n', bestK);
  fprintf('Accuracy: %.2f%%\n', accuracy * 100);
  fprintf('Precision: %.2f\n', mean(precision));
  fprintf('Recall: %.2f\n', mean(recall));
  fprintf('F1 Score: %.2f\n', mean(f1Scores));
  
  % Visualize the confusion matrix
  confusionchart(confMat, diseaseCategories);
