% 1. get data
d = csvread("train_new.csv");
X=d(:,2:end);
y=d(:,1);

% 2. Set up X and theta
X = mapFeature(X);
%initial_theta = zeros(size(X,2),1);

% 3. setup NN
X = X(:, 2:end); % Get rid of the bias column since we'll add that in later
input_layer_size  = size(X, 2);
hidden_layer_size = 15;   
num_labels = 1;
nn_params = [rand((1 + input_layer_size) * hidden_layer_size, 1) ; rand((hidden_layer_size + 1)*num_labels, 1)]; % Theta1 and Theta2

% Feed_Forward

fprintf('\nFeedforward Using Neural Network ...\n')

lambda = 4;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
                   

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 800);

%  You should also try different values of lambda


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                  D = csvread("test_modified.csv");
                  E=mapFeature(D);
                  E=E(:,2:end);
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pred = predict(Theta1, Theta2, E);
save prediction1.txt pred;
