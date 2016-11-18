fprintf('\nHere we train an RBM with Binary inputs (MNIST datastet).\n');

% LOAD DATASET
load('caltech101_silhouettes_28_split1.mat');

bestClassErr = 2;
bestLearningRate = 0;
bestBatchSize = 0;

for learningRate = 0.01:0.01:0.1
    for batchSize = 100:25:500
        [nObs,nVis] = size(train_data);

        nHid = 100; % 500 HIDDEN UNITS

        % DEFINE A MODEL ARCHITECTURE
        arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

        % GLOBAL OPTIONS
        arch.opts = {'verbose', 1, ...
                 'lRate', 0.1, ...
                'nEpoch', 100, ...
                'batchSz', 100, ...
                'nGibbs', 1};

        % INITIALIZE RBM
        r = rbm(arch);

        % TRAIN THE RBM
        r = r.train(train_data,single(train_labels));

        [~,classErr,misClass] = r.classify(test_data, single(test_labels));
        
        if (classErr < bestClassErr) 
            bestClassErr = classErr;
            bestLearningRate = learningRate;
            bestBatchSize = batchSize;
        end
        fprintf('\nDone a batch size.\n');
    end
    fprintf('\nDone a learning rate.\n');
end

fprintf('\nDone everything.\n');

%[nObs,nVis] = size(train_data);

%nHid = 100; % 500 HIDDEN UNITS

% DEFINE A MODEL ARCHITECTURE
%arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

% GLOBAL OPTIONS
%arch.opts = {'verbose', 1, ...
%		 'lRate', 0.1, ...
%		'momentum', 0.5, ...
%		'nEpoch', 10, ...
%		'wPenalty', 0.02, ...
%		'batchSz', 100, ...
%		'beginAnneal', 10, ...
%		'nGibbs', 1, ...
%		'sparsity', .01, ...
%		'varyEta',7, ...
%		'displayEvery', 20};
%  		'visFun', @visBinaryRBMLearning};

% GLOBAL OPTIONS
%arch.opts = {'verbose', 1, ...
%		 'lRate', 0.1, ...
%		'nEpoch', 10, ...
%		'batchSz', 100, ...
%		'nGibbs', 1};
%  		'visFun', @visBinaryRBMLearning};

% INITIALIZE RBM
%r = rbm(arch);

% TRAIN THE RBM
%r = r.train(train_data,single(train_labels));

%[~,classErr,misClass] = r.classify(test_data, single(test_labels));


%misClass = test_data(misClass,:);
%clf; visWeights(misClass',0,[0 1]); title(sprintf('Missclassified -- Error=%1.2f %%',classErr*100));

%nVis = 100;
%figure; visWeights(r.W(:,1:nVis));
%title('Sample of RBM Features');